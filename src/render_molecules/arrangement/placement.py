"""
./src/render_molecules/arrangement/placement.py

python -m src.render_molecules.arrangement.placement

Handles molecule placement in real-world environment, as well as cleanup steps, making it not only
physically plausible, but as accurate as possible without integration of time. This means taking
into account IMFs, force fields and more to create accurate images.
"""

import math
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from src.render_molecules.arrangement.geometry import (
    calculate_center_of_mass,
    check_instance_overlap,
    compute_bbox_center,
    compute_bounding_sphere_radius,
    get_rotation_matrix,
    point_in_bounds,
    radians,
)
from src.render_molecules.arrangement.scene_state import (
    MoleculeInstance,
    MoleculeTemplate,
    ObjectState,
)
from src.utils.type_annotations import Matrix3x1, Matrix3x3


@dataclass
class SpatialGrid:
    """
    Uniform 3D grid for fast neighbor lookup during placement.
    Each cell stores the IDs of instances whose center falls in that cell.
    Cell size is set to 2x the largest template bounding sphere radius, guaranteeing
    that any two overlapping molecules have centers in the same or adjacent cells.
    """

    cell_size: float
    origin: np.ndarray  # Min corner of the world box, shape (3,)
    nx: int
    ny: int
    nz: int
    cells: dict[tuple[int, int, int], list[int]] = field(default_factory=dict)

    def cell_of(self, position: np.ndarray) -> tuple[int, int, int]:
        """
        Returns the (i, j, k) cell index for a world-space position.

        Args:
            position (np.ndarray): World-space position of shape (3,).

        Returns:
            tuple[int, int, int]: Cell index (i, j, k).
        """
        offset = position - self.origin
        i = int(np.floor(offset[0] / self.cell_size))
        j = int(np.floor(offset[1] / self.cell_size))
        k = int(np.floor(offset[2] / self.cell_size))
        # Clamp to grid bounds to handle positions exactly on the max edge
        i = max(0, min(i, self.nx - 1))
        j = max(0, min(j, self.ny - 1))
        k = max(0, min(k, self.nz - 1))
        return (i, j, k)

    def insert(self, instance_id: int, position: np.ndarray) -> None:
        """
        Registers an instance in the cell containing its world-space position.

        Args:
            instance_id (int): Unique instance ID to store.
            position (np.ndarray): World-space position of shape (3,).
        """
        cell = self.cell_of(position)
        if cell not in self.cells:
            self.cells[cell] = []
        self.cells[cell].append(instance_id)

    def neighbors(self, position: np.ndarray) -> list[int]:
        """
        Returns all instance IDs in the 27-cell neighborhood around a position.

        Args:
            position (np.ndarray): World-space position of shape (3,).

        Returns:
            list[int]: Instance IDs of all molecules in neighboring cells.
        """
        ci, cj, ck = self.cell_of(position)
        result: list[int] = []
        for di, dj, dk in product((-1, 0, 1), repeat=3):
            cell = (ci + di, cj + dj, ck + dk)
            result.extend(self.cells.get(cell, []))
        return result


def build_spatial_grid(object_state: ObjectState) -> SpatialGrid:
    """
    Builds a SpatialGrid sized to the object's bounding box with cell size
    equal to 2x the largest template bounding sphere radius.

    Args:
        object_state (ObjectState): Scene state containing templates and world bounds.

    Returns:
        SpatialGrid: Empty grid ready for insertion.
    """
    max_radius = max(
        compute_bounding_sphere_radius(template=t)
        for t in object_state.templates.values()
    )
    cell_size = 2.0 * max_radius

    all_corners = np.vstack([object_state.box_bottom, object_state.box_top])
    min_corner = np.min(all_corners, axis=0)
    max_corner = np.max(all_corners, axis=0)
    dims = max_corner - min_corner

    nx = max(1, math.ceil(dims[0] / cell_size))
    ny = max(1, math.ceil(dims[1] / cell_size))
    nz = max(1, math.ceil(dims[2] / cell_size))

    return SpatialGrid(
        cell_size=cell_size,
        origin=min_corner,
        nx=nx,
        ny=ny,
        nz=nz,
    )


class PlacementConfig:
    """
    Dataclass that controls all aspects of molecule placement.
    Must be instantiated before calling place_molecules().
    """

    seed: (
        int | None
    )  # Random seed for deterministic placement. Set to a fixed value for reproducible results
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
    frontier_radius: float  # How close to an anchor to sample new candidates
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
    template: MoleculeTemplate,
    instance_id: int,
    position: Matrix3x1,
    rotation: Matrix3x3,
    velocity: np.ndarray | None = None,
) -> MoleculeInstance:
    """
    Creates a molecule instance given its qualities.

    Args:
        template_id (int): Unique template ID
        template (MoleculeTemplate): Molecule template used for COM correction
        instance_id (int): Unique instance ID
        position (np.ndarray): Desired world-space COM position vector
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

    local_com = calculate_center_of_mass(template=template)
    translation = position - (rotation @ local_com)

    vel = np.zeros(3, dtype=float) if velocity is None else velocity

    return MoleculeInstance(
        template_id=template_id,
        position=translation.astype(float),
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
        key for key, value in object_state.templates.items() if value is template
    ]
    if len(template_id) != 1:
        raise ValueError(
            f"Found {len(template_id)} appearances for {template.name}, expected 1 (whoopsies!)"
        )
    template_id = template_id[0]
    object_center = compute_bbox_center(
        object_state.box_bottom, object_state.box_top
    )  # Position
    rotation_matrix = sample_random_rotation(rng=rng)
    instance_id = 0
    return create_instance(
        template_id=template_id,
        template=template,
        instance_id=instance_id,
        position=object_center,
        rotation=rotation_matrix,
        velocity=None,
    )


def sample_candidate_pose(
    anchor_instance: MoleculeInstance, frontier_radius: float, rng: np.random.Generator
) -> tuple[Matrix3x1, Matrix3x3]:
    """
    Decides where to put a new molecule. Generates a random position somewhere in the frontier radius with a random rotation.

    Args:
        anchor_instance (MoleculeInstance): The frontier from which to sample.
        frontier_radius (float): Radius from CoM of anchor instance in which the new molecule will be sampled.
        rng (np.random.Generator): PRNG generator instance

    Returns:
        tuple[Matrix3x1, Matrix3x3]: position in world space, rotation matrix
    """
    direction = rng.standard_normal(3)
    direction /= np.linalg.norm(direction)
    distance = rng.uniform(0.0, frontier_radius)
    delta = direction * distance

    position = anchor_instance.position + delta

    return position, sample_random_rotation(rng=rng)


def check_placement(
    candidate_position: np.ndarray,
    candidate_rotation: np.ndarray,
    template: MoleculeTemplate,
    template_id: int,
    object_state: ObjectState,
    config: PlacementConfig,
    grid: SpatialGrid,
) -> bool:
    """
    Checks whether a candidate placement is valid: inside bounds and not overlapping
    any existing instance. Uses the spatial grid to avoid checking all n instances.

    Args:
        candidate_position (np.ndarray): Candidate world-space COM position, shape (3,).
        candidate_rotation (np.ndarray): Candidate rotation matrix, shape (3, 3).
        template (MoleculeTemplate): Template of the molecule being placed.
        template_id (int): ID of the template in object_state.templates.
        object_state (ObjectState): Current scene state.
        config (PlacementConfig): Placement configuration.
        grid (SpatialGrid): Spatial grid of already-placed instances.

    Returns:
        bool: True if the placement is valid, False if rejected.
    """
    if config.require_in_bounds:
        all_corners = np.vstack([object_state.box_bottom, object_state.box_top])
        min_corner = np.min(all_corners, axis=0)
        max_corner = np.max(all_corners, axis=0)
        if not point_in_bounds(candidate_position, (min_corner, max_corner)):
            return False

    if config.require_no_overlap:
        candidate = create_instance(
            template_id=template_id,
            template=template,
            instance_id=-1,  # Temporary; not registered in object_state
            position=candidate_position,
            rotation=candidate_rotation,
        )
        for instance_id in grid.neighbors(candidate_position):
            existing_instance = object_state.instances[instance_id]
            existing_template = object_state.templates[existing_instance.template_id]
            if check_instance_overlap(
                template_1=template,
                template_2=existing_template,
                instance_1=candidate,
                instance_2=existing_instance,
            ):
                return False

    return True

