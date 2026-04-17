"""
./src/seer_app.py

python -m src.seer_app

Main Seer app maker. Amalgamates everything into a single runnable file.
"""

from direct.showbase.ShowBase import ShowBase

from src.video_processing.environment import (
    AGGREGATION_PATH,
    RoomState,
    decrease_fov,
    env_setup,
    increase_fov,
    mouse_look,
    move,
)


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

    def _on_wheel_down(self) -> None:
        """
        On zoom out.
        """
        decrease_fov(self.room_state)


if __name__ == "__main__":
    app = SeerApp(aggregation_path=AGGREGATION_PATH, debug=False)
    app.run()
