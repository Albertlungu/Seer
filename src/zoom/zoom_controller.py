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
    LOG_FADE_END,
    LOG_FADE_START,
    LOG_MOL_DISTANCE,
    LOG_ROOM_DISTANCE,
    ROOM_REFERENCE_DISTANCE,
    SCROLL_STEP_SIZE,
)


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
    room_alpha: float = 1.0
    mol_alpha: float = 0.0
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


def compute_fade(
    log_distance: float, fade_start: float, fade_end: float
) -> tuple[float, float]:
    """
    Returns (room_alpha, mol_alpha) for the current log-distance. Room alpha is 1.0 when above fade_start,
    0.0 below fade_end, molecules are inverse.

    Args:
        log_distance (float): Current log10(camera_distance_metres).
        fade_start (float): Log10 distance where room begins fading (closer to 0)
        fade_end (float): Log10 distance where room is fully gone (more negative)

    Returns:
        tuple[float, float]: Tuple of (room_alpha, mol_alpha), each in [0.0, 1.0]
    """
    if log_distance >= fade_start:
        return 1.0, 0.0
    if log_distance <= fade_end:
        return 0.0, 1.0
    t = (log_distance - fade_end) / (fade_start - fade_end)
    return t, 1.0 - t


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
