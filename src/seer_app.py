"""
./src/seer_app.py

python -m src.seer_app

Main Seer app maker. Amalgamates everything into a single runnable file.
"""

from direct.showbase.ShowBase import ShowBase
from panda3d.core import TransparencyAttrib

from src.video_processing.environment import (
    AGGREGATION_PATH,
    RoomState,
    env_setup,
    mouse_look,
    move,
)
from src.zoom.zoom_controller import (
    ZoomController,
    compute_clip_planes,
    compute_movement_speed,
)
from src.utils.constants import BASE_MOVEMENT_SPEED


class SeerApp(ShowBase):
    def __init__(
        self,
        room_state: RoomState | None = None,
        zoom_controller: ZoomController | None = None,
        aggregation_path: str | None = AGGREGATION_PATH,
        debug: bool = False,
    ) -> None:
        """
        Initialization of the SeerApp entrypoint class.

        Args:
            room_state (RoomState | None, optional): The current room state with all necessary default values. Defaults to None.
            zoom_controller (ZoomController | None, optional): The zoom controller class to manage transitions. Defaults to None.
            aggregation_path (str | None, optional): The path to the aggregations JSON. Defaults to AGGREGATION_PATH.
            debug (bool, optional): Whether to print debug statements or not. Defaults to False.

        Raises:
            ValueError: If the loader does not exist.
        """
        super().__init__()

        self.disableMouse()

        self.room_root = self.render.attachNewNode("room_root")
        self.mol_root = self.render.attachNewNode("mol_root")
        self.room_root.setTransparency(TransparencyAttrib.MAlpha)
        self.mol_root.setTransparency(TransparencyAttrib.MAlpha)

        if room_state is None:
            room_state = RoomState(window=self.win, camera=self.camLens)

        self.zoom_controller = zoom_controller if zoom_controller else ZoomController()

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

        self.taskMgr.add(self._mouse_look_task, "mouse-look")
        self.taskMgr.add(self._move_task, "move")
        self.taskMgr.add(self._zoom_update_task, "zoom-update")

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
        On zoom in
        """
        self.zoom_controller.on_scroll(-1)

    def _on_wheel_down(self) -> None:
        """
        On zoom out
        """
        self.zoom_controller.on_scroll(1)

    def _zoom_update_task(self, task):
        self.zoom_controller.update()
        state = self.zoom_controller.state

        if self.camLens is None:
            return task.cont

        near_plane, far_plane = compute_clip_planes(state.distance)
        self.camLens.setNear(near_plane)
        self.camLens.setFar(far_plane)

        self.room_root.setColorScale(1.0, 1.0, 1.0, state.room_alpha)
        self.mol_root.setColorScale(1.0, 1.0, 1.0, state.mol_alpha)

        self.move_speed = compute_movement_speed(
            distance=state.distance,
            base_speed=self.room_state.current_move_speed,
        )

        return task.cont


if __name__ == "__main__":
    app = SeerApp(aggregation_path=AGGREGATION_PATH, debug=False)
    app.run()
