"""
./src/render_molecules/arrangement/geometry.py

python -m src.render_molecules.arrangement.geometry

Houses:
- Local -> world coordinate transform helpers, which take in the molecule's conformer coordinates
  and apply a transformation matrix/vector to them to place them in teh real world.
- Rotation and translation composition
    - Math helpers to rotate and shift arrays of atoms
    - e.g. taking an instance position and an orientation/representation (such as Euler angles or
      Quaternions) and creating a transform matrix
- Molecule extent and axis-aligned bbox helpers
    - function that takes a molecule template and calculates its local bounding box (min, max,
      XYZ ranges) or rough radius
    - to make sure atoms don't spawn inside each other
"""

import numpy as np

from src.render_molecules.arrangement.scene_state import (
    MoleculeInstance,
    MoleculeTemplate,
)
from src.utils.constants import ELEMENT_MASSES


def compute_molecule_bbox(template: MoleculeTemplate) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the min/max bounding boxes of molecule in its local space

    Args:
        local_xyz (tuple[np.ndarray, np.ndarray, np.ndarray]): Local XYZ coordinates.

    Returns:
        tuple[np.ndarray, np.ndarray]: Contains min_bound (min_x, min_y, min_z) and max_bound (max_x, max_y, max_z) as a BBOX
    """
    local_xyz = template.local_xyz

    min_bound = np.array(np.min(local_xyz, axis=1))  # Axis=1 gives one minimum per XYZ
    max_bound = np.array(np.max(local_xyz, axis=1))

    return min_bound, max_bound


def compute_instance_bbox_world(
    instance: MoleculeInstance,
) -> tuple[np.ndarray, np.ndarray]:
    pass


def apply_transformation(
    template: MoleculeTemplate,
    position: np.ndarray,
    rotation_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies rotation then translation.

    Args:
        local_xyz (tuple[np.ndarray, np.ndarray, np.ndarray]): Local XYZ coordinates of a molecule
        position (np.ndarray): Translation vector of shape (3,): [tx, ty, tz]
        rotation_matrix (np.ndarray): Rotation matrix of shape (3, 3)
                                      [ cos(theta)  -sin(theta)   0 ]
                                      [ sin(theta)   cos(theta)   0 ]
                                      [    0             0        1 ]

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing transformed XYZ
    """
    local_xyz = template.local_xyz
    coords = np.vstack(
        local_xyz
    )  # np.vstack stacks a tuple of multiple elements into arrays
    world = rotation_matrix @ coords
    world = world + position.reshape(3, 1)
    return world[0], world[1], world[2]


def get_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Creates a 3x3 rotation matrix from Euler angles, in radians. This function may not be used,
    since Panda3D just takes the angles in .setHpr()

    Args:
        yaw (float): Twisting around vertical axis (Z)
        pitch (float): Rotation around side-side axis (Y)
        roll (float): Rotation around forward-backward axis (X)

    Returns:
        np.ndarray: Rotation matrix of shape (3, 3)
    """
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)

    # ZYX rotation: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    Rz = np.array(
        [
            [cz, -sz, 0],
            [sz, cz, 0],
            [0, 0, 1],
        ]
    )
    Ry = np.array(
        [
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy],
        ]
    )
    Rx = np.array(
        [
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx],
        ]
    )

    return Rz @ Ry @ Rx


def calculate_center_of_mass(
    template: MoleculeTemplate,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates center of mass of a molecule in its local space.

    Args:
        template (MoleculeTemplate): The molecule template

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: XYZ of the local center of mass (COM)
    """
    local_xyz = template.local_xyz
    elements = template.elements

    element_masses_used = np.array(
        [ELEMENT_MASSES.get(element) for element in elements]
    )

    total_mass = sum(element_masses_used)

    com_x = np.sum(element_masses_used * local_xyz[0]) / total_mass
    com_y = np.sum(element_masses_used * local_xyz[1]) / total_mass
    com_z = np.sum(element_masses_used * local_xyz[2]) / total_mass

    return com_x, com_y, com_z


def distance_between_points(
    start: np.ndarray,
    end: np.ndarray,
) -> np.floating:
    """
    Calculates the distance between two points in 3D space

    Args:
        start (np.ndarray): Starting point, shape (3,)
        end (np.ndarray): End point, shape (3,)

    Returns:
        np.floating: Distance
    """
    return np.linalg.norm(end - start)
