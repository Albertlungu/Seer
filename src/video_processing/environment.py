"""
./src/video_processing/environment.py

python -m src.video_processing.environment

Creates the 3D environment in which the user can move around.
"""

# pyright: reportAttributeAccessIssue=none
# pyright: reportOptionalMemberAccess=none

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from direct.showbase.Loader import Loader
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from panda3d.core import (
    GraphicsWindow,
    Lens,
    NodePath,
    Point3,
    WindowProperties,
    loadPrcFileData,
)

loadPrcFileData("", "load-file-type p3assimp")

AGGREGATION_PATH = "./data/vision_json/aggregations.json"


@dataclass
class RoomState:
    window: GraphicsWindow
    camera: Lens
    debug: bool = False
    mvt_key_states = {
        "w": False,
        "s": False,
        "a": False,
        "d": False,
    }
    mouse_locked: bool = True

    current_fov: float = 90.0
    current_move_speed: float = 1.5

    fov_delta: float = 2.0
    move_speed_delta: float = 0.5

    default_move_speed: float = 1.5
    default_sensitivity: float = 0.1
    default_hpr: tuple[float, float, float] = (0.0, 90.0, 0.0)
    default_fov: float = 90.0
    default_near: float = 0.01
    aggregation_path: str = AGGREGATION_PATH
    movement_commands: list[tuple[str, Callable, Any]] = field(
        default_factory=lambda: [
            # Movement
            ("w", set_key, ["w", True]),
            ("w-up", set_key, ["w", False]),
            ("a", set_key, ["a", True]),
            ("a-up", set_key, ["a", False]),
            ("s", set_key, ["s", True]),
            ("s-up", set_key, ["s", False]),
            ("d", set_key, ["d", True]),
            ("d-up", set_key, ["d", False]),
            # Speed
            ("shift-=", increase_speed, None),
            ("-", decrease_speed, None),
            # FOV
            ("wheel_up", decrease_fov, None),
            ("wheel_down", increase_fov, None),
            # Misc
            ("escape", toggle_mouse_lock, None),
        ]
    )


def set_key(room_state: RoomState, key: str, value: bool) -> None:
    """
    Record whether a movement key is pressed or released.

    Args:
        room_state (RoomState): State container holding key states.
        key (str): The movement key ("w", "a", "s", "d").
        value (bool): True if pressed, False if released.
    """
    room_state.mvt_key_states[key] = value


def increase_fov(room_state: RoomState) -> None:
    """
    Adjust the camera field of view by fov_delta.

    Args:
        room_state (RoomState): State container with current FOV and fov_delta.
    """
    fov = room_state.camera.getFov()
    new_fov = max(0.1, min(100, fov[0] + room_state.fov_delta))
    room_state.camera.setFov(new_fov)


def increase_speed(room_state: RoomState) -> None:
    """
    Adjust movement speed by move_speed_delta, clamped to [0.1, 10.0].

    Args:
        room_state (RoomState): State container with current speed and delta.
    """
    room_state.current_move_speed = max(
        0.1, min(10, room_state.current_move_speed + room_state.move_speed_delta)
    )


def decrease_speed(room_state: RoomState) -> None:
    """
    Decrease movement speed by move_speed_delta, clamped to [0.1, 10.0].

    Args:
        room_state (RoomState): State container with current speed and delta.
    """
    room_state.current_move_speed = max(
        0.1, min(10, room_state.current_move_speed - room_state.move_speed_delta)
    )


def decrease_fov(room_state: RoomState) -> None:
    """
    Decrease camera field of view by fov_delta.

    Args:
        room_state (RoomState): State container with current FOV and fov_delta.
    """
    fov = room_state.camera.getFov()
    new_fov = max(0.1, min(100, fov[0] - room_state.fov_delta))
    room_state.camera.setFov(new_fov)


def toggle_mouse_lock(room_state: RoomState) -> None:
    """
    Toggle mouse lock and cursor visibility.

    Args:
        room_state (RoomState): State container with window, camera, and lock flag.
    """
    room_state.mouse_locked = not room_state.mouse_locked
    props = WindowProperties()
    props.setCursorHidden(room_state.mouse_locked)
    room_state.window.requestProperties(props)


def env_setup(loader: Loader, parent: NodePath, room_state: RoomState) -> NodePath:
    """
    Load and configure the room environment.

    Applies rotation, centering, camera setup, and window properties.

    Args:
        loader (Loader): The loader object from ShowBase.
        parent (NodePath): The parent node to attach the environment to.
        room_state (RoomState): State container with camera, window, and defaults.

    Returns:
        NodePath: The loaded and configured environment node.
    """

    env: NodePath = loader.loadModel(
        "/Users/albertlungu/Local/GitHub/Seer/data/reconstructions/obj/albert_room.obj"
    )  # If there are issues, revert to absolute path
    env.reparentTo(parent)
    env.setHpr(room_state.default_hpr)
    env.setTwoSided(True)

    center: tuple[Point3, Point3] = env.getTightBounds()
    mid: Point3 = (center[0] + center[1]) / 2
    env.setPos(-mid.x, -mid.y, -mid.z)

    room_state.camera.setNear(room_state.default_near)
    room_state.camera.setFov(room_state.default_fov)

    props = WindowProperties()
    props.setCursorHidden(True)
    room_state.window.requestProperties(props)

    return env


def mouse_look(self, task):
    """
    The method to calculate the location of the actual camera.

    Args:
        task (Task): Task object automatically passed by taskMgr. Provides frame timing.
    """
    if (
        self.mouse_locked and self.mouseWatcherNode.hasMouse()
    ):  # Check if mouse is in the window
        # Get window center in pixels
        cx = self.win.getXSize() // 2
        cy = self.win.getYSize() // 2

        # Current mouse position (normalized coordinates)
        nx = self.mouseWatcherNode.getMouseX()
        ny = self.mouseWatcherNode.getMouseY()
        # Current mouse position in pixels (converts normalized)
        px = round((nx + 1) * 0.5 * self.win.getXSize())
        py = round((1 - (ny + 1) * 0.5) * self.win.getYSize())

        # Distance from center
        dx = px - cx
        dy = cy - py  # Reversed for natural mouse movement

        # Dead zone to ignore sub-pixel noise
        if abs(dx) > 1 or abs(dy) > 1:
            self.heading -= dx * self.sensitivity
            self.pitch = max(
                -80, min(80, self.pitch + dy * self.sensitivity)
            )  # Clamps the vertical degrees to 80 degrees horizontally and vertically
            self.camera.setHpr(
                self.heading, self.pitch, 0
            )  # Applies the rotation to the camera

            # Makes sure the cursor is always centered at the middle of the screen
            self.win.movePointer(0, cx, cy)
            # After each frame, move the mouse back to center
    return task.cont


def move(self, task):
    """
    Moves the camera based on the WASD inputs

    Args:
        task (Task): Task object.
    """
    dt = globalClock.getDt()  # Number of seconds since last frame so that movement speed is consistent across monitors

    if self.keys["w"]:
        self.camera.setY(self.camera, self.move_speed * dt)
    if self.keys["s"]:
        self.camera.setY(self.camera, -self.move_speed * dt)
    if self.keys["a"]:
        self.camera.setX(self.camera, -self.move_speed * dt)
    if self.keys["d"]:
        self.camera.setX(self.camera, self.move_speed * dt)

    return task.cont


class Room(ShowBase):
    def __init__(self, debug: bool = False, aggregation_path: str | None = None):
        super().__init__()

        self.disableMouse()
        self.debug = debug
        if aggregation_path:
            with open(aggregation_path, "rb") as f:
                self.aggregations = json.load(f)
        if not self.loader:
            raise ValueError("Error; loader DNE")
        self.environ = self.loader.loadModel(
            "/Users/albertlungu/Local/GitHub/Seer/data/reconstructions/obj/albert_room.obj"
        )  # Loads the 3D model

        self.sensitivity = (
            0.1  # Sens in degrees of cam rotation per pixel of mouse movement
        )
        self.heading = 0  # Left/Right rotation
        self.pitch = 0  # Up/Down tilt

        # Register mouse_look as a per-frame task
        self.taskMgr.add(self.mouse_look, "mouse-look")
        # taskMgr is the task manager
        # It makes a function be called every frame to the main loop
        # Args:
        #   class: function to call
        #   string: name string

        self.mouse_locked = True

        # Bind movement commands from RoomState
        room_state = RoomState(
            window=self.win,
            camera=self.camLens,
            debug=debug,
            aggregation_path=aggregation_path or AGGREGATION_PATH,
        )
        for key, func, args in room_state.movement_commands:
            if args is not None:
                self.accept(key, func, args)
            else:
                self.accept(key, func)

        self.move_speed = 1.5  # In units/s
        self.delta_speed = 0.5  # By how much to change speed
        # self.accept(
        #     "shift-=", self.control_speed, [self.delta_speed]
        # )  # e.g. "+" to increase speed
        # self.accept("-", self.control_speed, [-self.delta_speed])  # - to decrease
        self.keys = {
            "w": False,
            "s": False,
            "a": False,
            "d": False,
        }  # Defining the keys used
        # self.accept("w", self.set_key, ["w", True])
        # self.accept("w-up", self.set_key, ["w", False])
        # self.accept("a", self.set_key, ["a", True])
        # self.accept("a-up", self.set_key, ["a", False])
        # self.accept("s", self.set_key, ["s", True])
        # self.accept("s-up", self.set_key, ["s", False])
        # self.accept("d", self.set_key, ["d", True])
        # self.accept("d-up", self.set_key, ["d", False])

        self.taskMgr.add(self.move, "move")

        # self.delta_zoom = 2  # By how many degrees to change FOV

        # self.accept("wheel_up", self.zoom, [-self.delta_zoom])
        # self.accept("wheel_down", self.zoom, [self.delta_zoom])


if __name__ == "__main__":
    app = Room(
        True,
        aggregation_path=AGGREGATION_PATH,
    )
    app.show_bbox()
    app.run()
