"""
./src/dynamics/frustum_culler.py

Camera frustum culling for selecting which molecules to simulate.
"""

import numpy as np

from src.render_molecules.arrangement.geometry import (
    calculate_environment_center_of_mass,
)
from src.render_molecules.arrangement.scene_state import ObjectState


def get_active_instances(
    camera_pos: np.ndarray,
    camera_forward: np.ndarray,
    fov_degrees: float,
    object_state: ObjectState,
    margin_factor: float = 1.5,
) -> list[int]:
    """
    Returns instance IDs whose centres of mass fall within the camera's
    view cone (plus a margin for smooth simulation entry).

    Args:
        camera_pos: Camera world position, shape (3,).
        camera_forward: Camera forward unit vector, shape (3,).
        fov_degrees: Horizontal field of view in degrees.
        object_state: Current object state with templates and instances.
        margin_factor: Multiplier on FOV to include molecules slightly outside view.

    Returns:
        List of instance IDs within the expanded view cone.
    """
    half_angle_rad = np.radians(fov_degrees * margin_factor / 2.0)
    cos_limit = np.cos(half_angle_rad)
    active: list[int] = []

    for iid, inst in object_state.instances.items():
        tmpl = object_state.templates[inst.template_id]
        com = calculate_environment_center_of_mass(template=tmpl, instance=inst)
        to_mol = com - camera_pos
        dist = float(np.linalg.norm(to_mol))

        if dist < 1e-15:
            active.append(iid)
            continue

        cos_angle = float(np.dot(to_mol / dist, camera_forward))
        if cos_angle >= cos_limit:
            active.append(iid)

    return active
