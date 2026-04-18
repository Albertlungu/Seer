"""
./src/seer_app.py

python -m src.seer_app

Main Seer app maker. Amalgamates everything into a single runnable file.
"""

from typing import cast

from direct.showbase.ShowBase import ShowBase
from panda3d.core import Point3

from src.utils.constants import FINAL_AGGREGATED
from src.utils.json_io import load_json
from src.video_processing.environment import (
    AGGREGATION_PATH,
    RoomState,
    decrease_fov,
    env_setup,
    increase_fov,
    mouse_look,
    move,
)
from src.zoom.raycast_picker import RaycastPicker


class SeerApp(ShowBase):
    def __init__(
        self,
        room_state: RoomState | None = None,
        aggregation_path: str | None = AGGREGATION_PATH,
        debug: bool = False,
    ) -> None:
        """
        Initialization of the SeerApp entrypoint class.

        Args:
            room_state (RoomState | None, optional): The current room state with all necessary default values. Defaults to None.
            aggregation_path (str | None, optional): The path to the aggregations JSON. Defaults to AGGREGATION_PATH.
            debug (bool, optional): Whether to print debug statements or not. Defaults to False.

        Raises:
            ValueError: If the loader does not exist.
        """
        super().__init__()

        self.disableMouse()

        self.room_root = self.render.attachNewNode("room_root")
        self.room_root.show()

        if room_state is None:
            room_state = RoomState(window=self.win, camera=self.camLens)

        self.room_state = room_state
        self.keys = self.room_state.mvt_key_states
        self.move_speed = self.room_state.current_move_speed
        self.mouse_locked = self.room_state.mouse_locked
        self.sensitivity = self.room_state.default_sensitivity
        self.heading = 0.0
        self.pitch = 0.0

        if aggregation_path:
            self.room_state.aggregation_path = aggregation_path
        self.room_state.debug = debug
        self.room_state.camera = self.camLens
        self.room_state.window = self.win

        if self.loader:
            self.room_geo = env_setup(
                loader=self.loader, parent=self.room_root, room_state=self.room_state
            )
        else:
            raise ValueError("Missing loader.")

        self.room_data = load_json(FINAL_AGGREGATED)
        self.room_picker = RaycastPicker(
            camera_node=self.camNode, target_root=self.room_geo
        )
        self.room_picker.mark_pickable(self.room_geo)
        self.object_bounds = self._build_object_bounds(self.room_data)

        self.taskMgr.add(self._mouse_look_task, "mouse-look")
        self.taskMgr.add(self._move_task, "move")

        for key, func, args in self.room_state.movement_commands:
            if args is not None:
                self.accept(key, func, [self.room_state, *args])
            else:
                self.accept(key, func, [self.room_state])

        self.accept("wheel_up", self._on_wheel_up)
        self.accept("wheel_down", self._on_wheel_down)

    def _mouse_look_task(self, task):
        self.mouse_locked = self.room_state.mouse_locked
        return mouse_look(self, task)

    def _move_task(self, task):
        return move(self, task)

    def _on_wheel_up(self) -> None:
        """
        On zoom in.
        """
        increase_fov(self.room_state)
        self._clear_target_lock()

    def _on_wheel_down(self) -> None:
        """
        On zoom out.
        """
        decrease_fov(self.room_state)
        if self.room_state.molecular_mode:
            self._lock_target_from_center()

    def _build_object_bounds(
        self, data: dict
    ) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
        bounds: dict[
            str, tuple[tuple[float, float, float], tuple[float, float, float]]
        ] = {}

        for object_key, object_data in data.items():
            corners = object_data["corners"]["bottom"] + object_data["corners"]["top"]
            mins = cast(
                tuple[float, float, float],
                tuple(
                    min(float(point[index]) for point in corners) for index in range(3)
                ),
            )
            maxs = cast(
                tuple[float, float, float],
                tuple(
                    max(float(point[index]) for point in corners) for index in range(3)
                ),
            )
            bounds[object_key] = (mins, maxs)

        return bounds

    def _find_object_key_for_point(self, point: Point3) -> str:
        candidate_key: str | None = None
        candidate_distance: float | None = None

        for object_key, (mins, maxs) in self.object_bounds.items():
            if all(
                mins[index] <= coord <= maxs[index]
                for index, coord in enumerate((point.x, point.y, point.z))
            ):
                return object_key

            center = tuple((mins[index] + maxs[index]) * 0.5 for index in range(3))
            distance = sum(
                (coord - center[index]) ** 2
                for index, coord in enumerate((point.x, point.y, point.z))
            )
            if candidate_distance is None or distance < candidate_distance:
                candidate_distance = distance
                candidate_key = object_key

        if candidate_key is None:
            raise RuntimeError("No object bounds were available for raycast locking.")

        return candidate_key

    def _lock_target_from_center(self) -> None:
        hit = self.room_picker.pick_center()
        if hit is None:
            return

        _hit_node, hit_point = hit
        object_key = self._find_object_key_for_point(hit_point)
        self.room_state.target_point = hit_point
        self.room_state.target_object_key = object_key
        self.room_state.target_locked = True

    def _clear_target_lock(self) -> None:
        self.room_state.target_point = None
        self.room_state.target_object_key = None
        self.room_state.target_locked = False


if __name__ == "__main__":
    app = SeerApp(aggregation_path=AGGREGATION_PATH, debug=False)
    app.run()
