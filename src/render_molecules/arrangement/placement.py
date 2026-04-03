"""
./src/render_molecules/arrangement/placement.py

python -m src.render_molecules.arrangement.placement

Handles molecule placement in real-world environment, as well as cleanup steps, making it not only
physically plausible, but as accurate as possible without integration of time. This means taking
into account IMFs, force fields and more to create accurate images.
"""

import numpy as np

from src.render_molecules.arrangement.geometry import (
    compute_bbox_center,
    get_rotation_matrix,
    radians,
)
from src.render_molecules.arrangement.scene_state import (
    MoleculeInstance,
    MoleculeTemplate,
    ObjectState,
)
from src.utils.type_annotations import Matrix3x3


class PlacementConfig:
    seed: int | None
    jitter_scale: (
        float  # How much a candidate molecule is nudged randomly when sampling pos
    )

    max_seed_attempts: int  # num tries for placing the first molecule
    max_candidate_attempts: (
        int  # num tries to place one non-seed molecule before giving up
    )
    max_total_attempts: int

    overlap_safety_factor: float
    min_center_distance: float | None

    use_frontier: bool = True
    frontier_radius: float
    frontier_max_rejections_per_anchor: int  # Count how many candidate placements fall near each anchor in the frontier  # If reached, stop using anchor or cool down (too crowded)
    frontier_max_size: int

    target_instance_count: int | None
    stop_when_target_met: bool

    enable_relaxation: bool  # Off/on switch for post-placement cleanup
    relaxation_passes: int  # How many cleanup rounds to run over placed molecules
    relaxation_step_size: float  # Max translation move per cleanup step
    relaxation_rotation_step: float  # Max rotation change per cleanup step

    require_in_bounds: bool
    require_no_overlap: bool = True


def create_instance(
    template_id: int,
    instance_id: int,
    position: np.ndarray,
    rotation: np.ndarray,
    velocity: np.ndarray | None = None,
) -> MoleculeInstance:
    """
    Creates a molecule instance given its qualities.

    Args:
        template_id (int): Unique template ID
        instance_id (int): Unique instance ID
        position (np.ndarray): Position vector
        rotation (np.ndarray): Rotation matrix
        velocity (np.ndarray | None, optional): Velocity vector. Defaults to None.

    Raises:
        ValueError: If position shape is not (3,)
        ValueError: If rotation shape is not (3, 3)

    Returns:
        MoleculeInstance: Molecule instance with the given qualities.
    """
    if position.shape != (3,):
        raise ValueError("Position must be of shape (3,)")
    if rotation.shape != (3, 3):
        raise ValueError("Rotation must be of shape (3, 3)")

    vel = np.zeros(3, dtype=float) if velocity is None else velocity

    return MoleculeInstance(
        template_id=template_id,
        position=position.astype(float),
        rotation=rotation.astype(float),
        velocity=vel.astype(float),
        id=instance_id,
    )


def sample_random_rotation(rng: np.random.Generator) -> Matrix3x3:
    """
    Gets a random rotation using an rng.

    Args:
        rng (np.random.Generator): rng generated from seed

    Returns:
        Matrix3x3: Random rotation matrix of shape (3, 3)
    """
    yaw, pitch, roll = rng.uniform(
        low=0,
        high=360,
        size=3,
    )
    yaw, pitch, roll = radians(yaw), radians(pitch), radians(roll)

    return get_rotation_matrix(yaw=yaw, pitch=pitch, roll=roll)


def place_seed_instance(
    object_state: ObjectState,
    template: MoleculeTemplate,
    rng: np.random.Generator,
) -> MoleculeInstance:
    """
    Creates the seed instance at the center of an object's bounding box.

    Args:
        object_state (ObjectState): The current object state dataclass containing all necessary info
        template (MoleculeTemplate): The target molecule template
        rng (np.random.Generator): PRNG generator instance

    Raises:
        ValueError: If there are more than one appearances for the same template in the object.

    Returns:
        MoleculeInstance: The translated and rotated seed molecule instance
    """
    template_id = [
        key for key, value in object_state.templates.items() if value == template
    ]
    if len(template_id) > 0:
        raise ValueError(
            f"Found {len(template_id)} appearances for {template.name}, expected 1 (whoopsies!)"
        )
    else:
        template_id = template_id[0]
    object_center = compute_bbox_center(
        object_state.box_bottom, object_state.box_top
    )  # Position
    rotation_matrix = sample_random_rotation(rng=rng)
    instance_id = 0
    return create_instance(
        template_id=template_id,
        instance_id=instance_id,
        position=object_center,
        rotation=rotation_matrix,
        velocity=None,
    )
