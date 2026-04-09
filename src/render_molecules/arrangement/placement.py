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


@dataclass
class PlacementConfig:
    """
    Dataclass that controls all aspects of molecule placement.
    Must be instantiated before calling place_molecules().
    """

    seed: int | None = 0
    jitter_scale: float = 0.0

    max_seed_attempts: int = 128
    max_candidate_attempts: int = 128
    max_total_attempts: int = 5000

    overlap_safety_factor: float = 1.0
    min_center_distance: float | None = None

    use_frontier: bool = True
    frontier_radius: float = 0.25
    frontier_max_rejections_per_anchor: int = 32
    frontier_max_size: int = 256

    target_instance_count: int | None = None
    stop_when_target_met: bool = True

    enable_relaxation: bool = False
    relaxation_passes: int = 0
    relaxation_step_size: float = 0.0
    relaxation_rotation_step: float = 0.0

    require_in_bounds: bool = True
    require_no_overlap: bool = True


def create_instance(
    template_id: int,
    object_state: ObjectState,
    instance_id: int,
    position: Matrix3x1,
    rotation: Matrix3x3,
    velocity: np.ndarray | None = None,
) -> MoleculeInstance:
    """
    Creates a molecule instance given its qualities.

    Args:
        template_id (int): Unique template ID
        object_state (ObjectState): Scene state used to look up the template for COM correction
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

    template = object_state.templates[template_id]
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


def sample_random_rotation(
    rng: np.random.Generator,
) -> tuple[Matrix3x3, tuple[float, float, float]]:
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

    return get_rotation_matrix(yaw=yaw, pitch=pitch, roll=roll), (yaw, pitch, roll)


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
    rotation_matrix = sample_random_rotation(rng=rng)[0]
    instance_id = 0
    return create_instance(
        template_id=template_id,
        object_state=object_state,
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

    return position, sample_random_rotation(rng=rng)[0]


def check_placement(
    candidate_position: np.ndarray,
    candidate_rotation: np.ndarray,
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
        template_id (int): ID of the template in object_state.templates.
        object_state (ObjectState): Current scene state.
        config (PlacementConfig): Placement configuration.
        grid (SpatialGrid): Spatial grid of already-placed instances.

    Returns:
        bool: True if the placement is valid, False if rejected.
    """
    template = object_state.templates[template_id]

    if config.require_in_bounds:
        all_corners = np.vstack([object_state.box_bottom, object_state.box_top])
        min_corner = np.min(all_corners, axis=0)
        max_corner = np.max(all_corners, axis=0)
        if not point_in_bounds(candidate_position, (min_corner, max_corner)):
            return False

    if config.require_no_overlap:
        candidate = create_instance(
            template_id=template_id,
            object_state=object_state,
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


def select_next_anchor(
    active_frontier: dict[int, int],
    config: PlacementConfig,
    rng: np.random.Generator,
) -> int:
    """
    Selects the next anchor from the active frontier.
    Filters out anchors that have hit their rejection limit, then picks randomly among the remaining candidates.

    Args:
        active_frontier (dict[int, int]): Maps instance ID to its rejection count.
        config (PlacementConfig): Placement configuration.
        rng (np.random.Generator): PRNG generator instance

    Raises:
        ValueError: If there are no more valid anchors in the frontier.

    Returns:
        int: Instance ID of the selected anchor
    """
    valid = [
        instance_id
        for instance_id, rejections in active_frontier.items()
        if rejections < config.frontier_max_rejections_per_anchor
    ]
    if not valid:
        raise ValueError("No valid anchors")
    return int(rng.choice(valid))


def schedule_next_molecule(
    target_counts: dict[int, int],
    placed_counts: dict[int, int],
    rng: np.random.Generator,
) -> int:
    """
    Selects which template to place next based on how far each template is from its target count.
    Templates furthest behind from their target counts are preferred.

    Args:
        target_counts (dict[int, int]): Maps template ID to target instance count.
        placed_counts (dict[int, int]): Maps template ID to currently placed count.
        rng (np.random.Generator): PRNG generator instance.

    Raises:
        ValueError: If all templates have met their target

    Returns:
        int: Template ID of next molecule to place.
    """
    deficit = {
        template_id: target_counts[template_id] - placed_counts.get(template_id, 0)
        for template_id in target_counts
    }
    remaining = {tid: d for tid, d in deficit.items() if d > 0}
    if not remaining:
        raise ValueError("All templates have met target counts")

    total = sum(remaining.values())
    weights = [remaining[tid] / total for tid in remaining]
    return int(rng.choice(list(remaining.keys()), p=weights))


def relax_overlaps():  # Stub for now
    pass


def place_molecules(
    object_state: ObjectState,
    config: PlacementConfig,
    target_counts: dict[int, int],
) -> ObjectState:
    """
    Main loop for placing molecules.

    Args:
        object_state (ObjectState): The object state container with templates and instances.
        config (PlacementConfig): The placement configuration.
        target_counts (dict[int, int]): Number of targets.

    Raises:
        ValueError: If the templates list in ObjectState is empty.

    Returns:
        ObjectState: The final object state, with the modified instances appended.
    """

    # --- Creation of persistent variables ---

    rng = np.random.default_rng(seed=config.seed)
    grid = build_spatial_grid(object_state=object_state)
    active_frontier: dict[int, int] = {}  # Instance id: rejection count
    placed_counts: dict[int, int] = {
        template_id: 0 for template_id in target_counts.keys()
    }
    next_instance_id = 1

    if not object_state.templates:
        raise ValueError("No templates available for placement")

    # --- Placement of seed template ---

    initial_template_id = schedule_next_molecule(
        target_counts=target_counts, placed_counts=placed_counts, rng=rng
    )
    initial_template = object_state.templates[initial_template_id]

    seed_instance = place_seed_instance(
        object_state=object_state,
        template=initial_template,
        rng=rng,
    )
    object_state.instances[0] = seed_instance

    active_frontier[0] = (
        0  # Adds seed instance to active frontier with rejection count zero
    )
    placed_counts[initial_template_id] += 1

    # --- Main loop ---
    for _ in range(config.max_total_attempts):
        # Pick which molecule type to place next and which anchor to grow from
        next_template_id = schedule_next_molecule(
            target_counts=target_counts, placed_counts=placed_counts, rng=rng
        )
        next_anchor_instance_id = select_next_anchor(
            active_frontier=active_frontier, config=config, rng=rng
        )

        # Sample a candidate position and rotation near the chosen anchor
        candidate_position, candidate_rotation = sample_candidate_pose(
            anchor_instance=object_state.instances[next_anchor_instance_id],
            frontier_radius=config.frontier_radius,
            rng=rng,
        )

        if check_placement(
            candidate_position=candidate_position,
            candidate_rotation=candidate_rotation,
            template_id=next_template_id,
            object_state=object_state,
            config=config,
            grid=grid,
        ):
            # Placement accepted: register the new instance
            new_instance = create_instance(
                template_id=next_template_id,
                object_state=object_state,
                instance_id=next_instance_id,
                position=candidate_position,
                rotation=candidate_rotation,
            )
            object_state.instances[next_instance_id] = new_instance
            grid.insert(instance_id=next_instance_id, position=candidate_position)

            # Add to frontier and trim if over the size limit
            active_frontier[next_instance_id] = 0
            if len(active_frontier) > config.frontier_max_size:
                worst = max(active_frontier, key=lambda iid: active_frontier[iid])
                del active_frontier[worst]

            placed_counts[next_template_id] += 1
            next_instance_id += 1

            # Stop early if all targets are met
            if all(placed_counts[tid] >= target_counts[tid] for tid in target_counts):
                break

        else:
            # Placement rejected: penalize the anchor that was tried
            active_frontier[next_anchor_instance_id] += 1

    if config.enable_relaxation:
        relax_overlaps()

    return object_state
