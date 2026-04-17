"""
./src/seer_app.py

python -m src.seer_app

Main Seer app maker. Amalgamates everything into a single runnable file.
"""

from direct.showbase.ShowBase import ShowBase

from src.video_processing.environment import (
    AGGREGATION_PATH,
    RoomState,
    env_setup,
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
        super().__init__()
        self.disableMouse()

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
                loader=self.loader, parent=self.render, room_state=self.room_state
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

    def _mouse_look_task(self, task):
        self.mouse_locked = self.room_state.mouse_locked
        return mouse_look(self, task)

    def _move_task(self, task):
        self.move_speed = self.room_state.current_move_speed
        return move(self, task)


if __name__ == "__main__":
    app = SeerApp(aggregation_path=AGGREGATION_PATH, debug=False)
    app.run()
