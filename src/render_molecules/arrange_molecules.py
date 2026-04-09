"""
./src/render_molecules/arrange_molecules.py

python -m src.render_molecules.arrange_molecules

Arrangement pipeline entrypoint module.

Serves as the top-level orchestration script for building realistic static molecular arrangements from aggregated JSON input.
"""

import argparse

import numpy as np
from direct.showbase.ShowBase import ShowBase

from src.render_molecules.arrangement.geometry import (
    calculate_center_of_mass,
    compute_bounding_sphere_radius,
)
from src.render_molecules.arrangement.placement import PlacementConfig, place_molecules
from src.render_molecules.arrangement.renderer import render_object_state
from src.render_molecules.arrangement.scene_state import (
    MoleculeInstance,
    MoleculeTemplate,
    ObjectState,
)
from src.utils.constants import FINAL_AGGREGATED
from src.utils.json_io import load_json
from src.utils.type_annotations import (
    AtomDetails,
    BondDetails,
    CoordsDetails,
    SimDetails,
)


def _extract_target_count(molecule_details: dict) -> int:
    """
    Extracts target copy count for one molecule entry.

    Args:
        molecule_details (dict): Composition entry for one molecule.

    Returns:
        int: Requested count if present, otherwise 1.
    """
    for key in ("count", "quantity", "target_count", "instances", "copies"):
        value = molecule_details.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return 1


def build_object_state(
    object_key: str | None = None,
) -> tuple[ObjectState, dict[int, int]]:
    """
    Builds an ObjectState and per-template target counts from aggregated input.

    Args:
        object_key (str | None): Optional object key to load. If None, uses first object.

    Raises:
        ValueError: If aggregated input is empty, key is missing, or sim_details is incomplete.

    Returns:
        tuple[ObjectState, dict[int, int]]: Scene state and target counts by template ID.
    """
    data = load_json(FINAL_AGGREGATED)
    if not data:
        raise ValueError("No objects found in aggregated input")

    selected_key = object_key if object_key is not None else next(iter(data.keys()))
    if selected_key not in data:
        raise ValueError(f"Object '{selected_key}' was not found in aggregated input")

    object_details = data[selected_key]
    object_state = ObjectState(
        object_key=selected_key,
        object_name=object_details.get("name", selected_key),
        instance_id=selected_key,
        display_name=selected_key.replace("_", " ").title(),
        templates={},
        instances={},
        box_bottom=np.array(object_details["corners"]["bottom"], dtype=float),
        box_top=np.array(object_details["corners"]["top"], dtype=float),
        rng_seed=1,
    )

    target_counts: dict[int, int] = {}
    for i, (molecule_name, molecule_details) in enumerate(
        object_details["composition"].items()
    ):
        sim_details: SimDetails | None = molecule_details.get("sim_details")
        if sim_details is None:
            raise ValueError(
                f"Missing sim_details for molecule '{molecule_name}' in object '{selected_key}'"
            )

        atom_details: AtomDetails | None = sim_details.get("atoms")
        bond_details: BondDetails | None = sim_details.get("bonds")
        coords = sim_details.get("coords")
        if not atom_details or not bond_details or not coords:
            raise ValueError(
                f"Empty sim_details for molecule '{molecule_name}' in object '{selected_key}'"
            )

        coords_details: CoordsDetails = coords[0]
        conformer = coords_details["conformers"][0]
        molecule_template = MoleculeTemplate(
            name=molecule_name,
            aids=np.array(atom_details["aid"], dtype=int),
            elements=np.array(atom_details["element"], dtype=int),
            local_xyz=(
                np.array(conformer["x"], dtype=float),
                np.array(conformer["y"], dtype=float),
                np.array(conformer["z"], dtype=float),
            ),
            bonds_aid1=np.array(bond_details["aid1"], dtype=int),
            bonds_aid2=np.array(bond_details["aid2"], dtype=int),
            bond_order=np.array(bond_details["order"], dtype=int),
        )

        object_state.templates[i] = molecule_template
        target_counts[i] = _extract_target_count(molecule_details)

    return object_state, target_counts


def _default_placement_config(
    seed: int, target_counts: dict[int, int], object_state: ObjectState
) -> PlacementConfig:
    """
    Creates the default placement configuration for full-object rendering.

    Args:
        seed (int): Random seed used by placement.
        target_counts (dict[int, int]): Target counts by template ID.
        object_state (ObjectState): Scene state used to compute molecule sizes.

    Returns:
        PlacementConfig: Placement configuration instance.
    """
    target_total = int(sum(target_counts.values()))
    max_radius = max(
        compute_bounding_sphere_radius(template=t)
        for t in object_state.templates.values()
    )
    # Molecules are sampled from [min_center_distance, frontier_radius] away from the
    # anchor COM. min_center_distance = 2*max_radius ensures bounding spheres don't
    # overlap at the sampling stage. The 0.5 Å gap on top gives a small surface clearance.
    return PlacementConfig(
        seed=seed,
        max_total_attempts=max(1000, target_total * 100),
        min_center_distance=2.0 * max_radius,
        frontier_radius=2.0 * max_radius + 0.5,
        target_instance_count=target_total,
        stop_when_target_met=True,
        require_in_bounds=True,
        require_no_overlap=True,
    )


def _initial_camera_pose(object_state: ObjectState) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes initial camera focus and position for first frame.

    Args:
        object_state (ObjectState): Arranged object state.

    Returns:
        tuple[np.ndarray, np.ndarray]: Focus point and camera position vectors.
    """
    corners = np.vstack([object_state.box_bottom, object_state.box_top])
    min_corner = np.min(corners, axis=0)
    max_corner = np.max(corners, axis=0)
    box_center = (min_corner + max_corner) * 0.5
    extent = float(np.linalg.norm(max_corner - min_corner))

    if not object_state.instances:
        focus = box_center
    else:
        centers: list[np.ndarray] = []
        for instance in object_state.instances.values():
            template = object_state.templates[instance.template_id]
            local_com = calculate_center_of_mass(template)
            world_com = (instance.rotation @ local_com) + instance.position
            centers.append(np.asarray(world_com, dtype=float).reshape(3))
        focus = np.mean(np.vstack(centers), axis=0)

    cam_distance = max(4.0, extent * 1.8)
    cam_height = max(1.5, extent * 0.6)
    camera_pos = np.array(
        [focus[0], focus[1] - cam_distance, focus[2] + cam_height],
        dtype=float,
    )
    return focus, camera_pos


class ArrangementApp(ShowBase):
    """Panda3D app that renders one arranged object state."""

    def __init__(self, object_state: ObjectState) -> None:
        """
        Initializes the Panda3D app and renders the arranged object.

        Args:
            object_state (ObjectState): Arranged object state to render.
        """
        super().__init__()
        self.setBackgroundColor(0.08, 0.08, 0.1, 1.0)
        focus, camera_pos = _initial_camera_pose(object_state)
        if self.camera is not None:
            self.camera.setPos(
                float(camera_pos[0]),
                float(camera_pos[1]),
                float(camera_pos[2]),
            )
            self.camera.lookAt(float(focus[0]), float(focus[1]), float(focus[2]))
        self.rendered_roots = render_object_state(
            base=self,
            parent=self.render,
            object_state=object_state,
        )


def main() -> None:
    """Parses CLI args, arranges molecules, and launches the renderer."""
    parser = argparse.ArgumentParser(description="Arrange and render one full object")
    parser.add_argument(
        "--object-key",
        type=str,
        default=None,
        help="Object key from final_aggregated.json. Defaults to first object.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed used by placement.",
    )
    args = parser.parse_args()

    object_state, target_counts = build_object_state(object_key=args.object_key)
    config = _default_placement_config(
        seed=args.seed, target_counts=target_counts, object_state=object_state
    )
    arranged_state = place_molecules(
        object_state=object_state,
        config=config,
        target_counts=target_counts,
    )

    app = ArrangementApp(object_state=arranged_state)
    app.run()


if __name__ == "__main__":
    main()
