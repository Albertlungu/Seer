"""
./src/zoom/zoom_controller.py

python -m src.zoom.zoom_controller

Logarithmic zoom state and per-frame derived quantities. Just the math.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal

import numpy as np

from src.utils.constants import (
    BASE_MOVEMENT_SPEED,
    LOG_MOL_DISTANCE,
    LOG_ROOM_DISTANCE,
    ROOM_REFERENCE_DISTANCE,
    SCROLL_STEP_SIZE,
)
from src.utils.type_annotations import Matrix3x1


class ViewMode(Enum):
    """
    Describes the possible states the user can be in.
    Room mode is when they are at the macro level.
    Transitioning is when they are in the process of zooming in.
    Molecular is when they are at the molecular level.
    """

    ROOM = auto()
    TRANSITIONING = auto()
    MOLECULAR = auto()


@dataclass
class ZoomState:
    log_distance: float = LOG_ROOM_DISTANCE
    target_point: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_object_key: str | None = None
    mode: ViewMode = ViewMode.ROOM
    distance: float = 10.0**LOG_ROOM_DISTANCE


def compute_clip_planes(distance: float) -> tuple[float, float]:
    """
    Returns (near, far) clip plane values for a given camera distance in metres.

    Args:
        distance (float): Positive camera distance from the zoom target.

    Returns:
        tuple[float, float]: Tuple of (near_plane, far_plane) in the same unit as distance.
    """
    near = max(distance * 0.001, 1e-15)
    far = max(distance * 1000.0, 10.0)
    return near, far


def compute_movement_speed(distance: float, base_speed: float) -> float:
    """
    Returns WASD movement speed scaled to current camera distance.

    Args:
        distance (float): Current camera distance from target in metres.
        base_speed (float): Speed in m/s at room scale.

    Returns:
        float: Scaled speed in m/s
    """
    return base_speed * (distance / ROOM_REFERENCE_DISTANCE)


def log_distance_after_scroll(
    current_log: float, scroll_direction: Literal[-1, 1], step_size: float
) -> float:
    """
    Advances log_distance by one scroll step.

    Args:
        current_log (float): Current log10 distance
        scroll_direction (Literal[-1, 1]): -1 to zoom in, +1 to zoom out.
        step_size (float): Size of one scroll step in units

    Returns:
        float: New log10 distance, clamped to [LOG_MOL_DISTANCE, LOG_ROOM_DISTANCE].
    """
    new_log = current_log + (scroll_direction * step_size)
    return max(LOG_MOL_DISTANCE, min(LOG_ROOM_DISTANCE, new_log))


class ZoomController:
    """
    Manages zoom state and computes per-frame derived values.

    Attributes:
        state: Mutable zoom state containing the current log-distance, target point, selected object key, and mode.
        _prev_mode: ViewMode from previous update call
        _target_locked: Whether the current zoom target has been set.
    """

    def __init__(self) -> None:
        self.state: ZoomState = ZoomState()
        self._prev_mode: ViewMode = ViewMode.ROOM
        self._target_locked: bool = False

    def on_scroll(self, direction: Literal[-1, 1]) -> None:
        """
        Handle one scroll tick

        Args:
            direction (Literal[-1, 1]): -1 for zoom in +1 for zoom out
        """
        self.state.log_distance = log_distance_after_scroll(
            current_log=self.state.log_distance,  # Gets from previous zoom tick
            scroll_direction=direction,
            step_size=SCROLL_STEP_SIZE,
        )

    def update(self) -> ViewMode:
        """
        Recompute all derived quantities from current log_distance.

        Returns:
            ViewMode: Current ViewMode (callers can check for transitions)
        """
        self.state.distance = 10**self.state.log_distance

        self._prev_mode = self.state.mode
        if self.state.log_distance >= LOG_ROOM_DISTANCE:
            self.state.mode = ViewMode.ROOM
        elif self.state.log_distance <= LOG_MOL_DISTANCE:
            self.state.mode = ViewMode.MOLECULAR
        else:
            self.state.mode = ViewMode.TRANSITIONING

        return self.state.mode

    def entered_transition(self) -> bool:
        """
        Checks if the user has changed from the room to the transition stage.

        Returns:
            bool: True on the frame in which the user enters transition, false otherwise.
        """
        return (
            self._prev_mode == ViewMode.ROOM
            and self.state.mode == ViewMode.TRANSITIONING
        )

    def exited_to_room(self) -> bool:
        """
        Checks if the user returns to room from transitioning.

        Returns:
            bool: True on the frame in which the user leaves transition and moves to room, false otherwise.
        """
        return (
            self._prev_mode == ViewMode.TRANSITIONING
            and self.state.mode == ViewMode.ROOM
        )

    def lock_target(self, point: Matrix3x1, object_key: str) -> None:
        """
        Sets the zoom target point and object key. Called once when the raycast runs.

        Args:
            point (Matrix3x1): Exact point at which the user is looking at.
            object_key (str): The unique identifier of each object.
        """
        self.state.target_point = point.copy()
        self.state.target_object_key = object_key
        self._target_locked = True

    def clear_target(self) -> None:
        """
        Reset the zoom target when returning to room scale.
        """
        self.state.target_object_key = None
        self._target_locked = False

    @property
    def target_locked(self) -> bool:
        return self._target_locked
