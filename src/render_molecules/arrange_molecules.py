"""
./src/render_molecules/arrange_molecules.py

python -m src.render_molecules.arrange_molecules

Arrangement pipeline entrypoint module.

Serves as the top-level orchestration script for building realistic static molecular arrangements from aggregated JSON input.
"""

import math

import numpy as np
from direct.showbase.ShowBase import ShowBase

from src.render_molecules.arrangement.placement import PlacementConfig, place_molecules
from src.render_molecules.arrangement.renderer import render_object_state
from src.render_molecules.arrangement.scene_state import (
    MoleculeTemplate,
    ObjectState,
)
from src.utils.constants import ANGSTROM_TO_METRES
from src.utils.json_io import load_json

_JSON_FILENAME = "final_aggregated.json"
_TARGET_COUNT_PER_TEMPLATE = 5
_RNG_SEED = 42


def build_templates_from_object(object_data: dict) -> dict[int, MoleculeTemplate]:
    """
    Extracts molecule templates from one object entry's composition dict.
    Skips molecules that lack sim_details.

    Args:
        object_data (dict): One object entry from the aggregated JSON.

    Returns:
        dict[int, MoleculeTemplate]: Map from template ID to template.
    """
    templates: dict[int, MoleculeTemplate] = {}
    template_id = 0

    for mol_name, mol_data in object_data["composition"].items():
        if "sim_details" not in mol_data:
            continue

        sd = mol_data["sim_details"]
        aids = np.array(sd["atoms"]["aid"], dtype=int)
        elements = np.array(sd["atoms"]["element"], dtype=int)

        conformer = sd["coords"][0]["conformers"][0]
        xs = np.array(conformer["x"], dtype=float)
        ys = np.array(conformer["y"], dtype=float)
        zs = np.array(conformer["z"], dtype=float)

        bonds = sd.get("bonds", {})
        bonds_aid1 = np.array(bonds.get("aid1", []), dtype=int)
        bonds_aid2 = np.array(bonds.get("aid2", []), dtype=int)
        bond_order = np.array(bonds.get("order", []), dtype=int)

        templates[template_id] = MoleculeTemplate(
            name=mol_name,
            aids=aids,
            elements=elements,
            local_xyz=(xs, ys, zs),
            bonds_aid1=bonds_aid1,
            bonds_aid2=bonds_aid2,
            bond_order=bond_order,
        )
        template_id += 1

    return templates


def build_object_state(
    object_key: str,
    object_data: dict,
    templates: dict[int, MoleculeTemplate],
    rng_seed: int,
) -> ObjectState:
    """
    Constructs an empty ObjectState from one JSON object entry.
    Derives world bounds from the object's corner arrays.

    Args:
        object_key (str): Canonical key from the JSON, e.g. "books".
        object_data (dict): The object entry dict.
        templates (dict[int, MoleculeTemplate]): Pre-built template map.
        rng_seed (int): Random seed for reproducible placement.

    Returns:
        ObjectState: Empty state container ready for placement.
    """
    corners = object_data["corners"]
    # JSON gives 4 points as [[x,y,z], ...] — transpose to (3, 4) as Matrix3x4
    # Convert from meters to Angstroms to match molecular coordinate system
    box_bottom = np.array(corners["bottom"], dtype=float).T / ANGSTROM_TO_METRES  # (3, 4)
    box_top = np.array(corners["top"], dtype=float).T / ANGSTROM_TO_METRES        # (3, 4)

    return ObjectState(
        object_key=object_key,
        object_name=object_data["name"],
        instance_id=object_key,
        display_name=object_data["name"],
        templates=templates,
        instances={},
        box_bottom=box_bottom,
        box_top=box_top,
        rng_seed=rng_seed,
    )


def run_arrangement() -> None:
    """
    Runs the full pipeline: load JSON, build templates, place molecules,
    build Panda3D scene, and start the app loop.
    """
    data = load_json(_JSON_FILENAME)

    object_key = next(iter(data))
    object_data = data[object_key]

    templates = build_templates_from_object(object_data)
    if not templates:
        raise RuntimeError(f"No valid templates found for object: {object_key!r}")

    object_state = build_object_state(
        object_key=object_key,
        object_data=object_data,
        templates=templates,
        rng_seed=_RNG_SEED,
    )

    target_counts = {tid: _TARGET_COUNT_PER_TEMPLATE for tid in templates}
    total_target = sum(target_counts.values())

    # Calculate frontier radius based on molecular scale
    # Use the maximum bounding sphere radius among all templates
    from src.render_molecules.arrangement.geometry import compute_bounding_sphere_radius
    max_radius = max(
        compute_bounding_sphere_radius(template) for template in templates.values()
    )
    # Set frontier radius to be several molecular diameters
    frontier_radius = max_radius * 6.0

    _bb_min = np.minimum.reduce(object_state.box_bottom.T)
    _bb_max = np.maximum.reduce(object_state.box_top.T)
    box_diag = float(np.linalg.norm(_bb_max - _bb_min))

    config = PlacementConfig(
        seed=_RNG_SEED,
        frontier_radius=frontier_radius,
        min_center_distance=max_radius * 3.5,  # Minimum separation: 1.75× molecular diameter for clear spacing
        max_total_attempts=total_target * 100,
        target_instance_count=total_target,
        stop_when_target_met=True,
        require_in_bounds=False,  # Molecules placed at origin, not constrained by physical bbox
        require_no_overlap=True,
    )

    object_state = place_molecules(
        object_state=object_state,
        config=config,
        target_counts=target_counts,
    )

    base = ShowBase()
    base.setBackgroundColor(0.05, 0.05, 0.08, 1.0)

    # Position camera to see the molecular cluster (not the entire box)
    if object_state.instances:
        # Calculate extent of placed molecules
        from src.render_molecules.arrangement.geometry import calculate_environment_center_of_mass

        positions = []
        for instance in object_state.instances.values():
            template = object_state.templates[instance.template_id]
            com = calculate_environment_center_of_mass(template=template, instance=instance)
            positions.append(com)

        positions_array = np.array(positions)
        mol_min = np.min(positions_array, axis=0)
        mol_max = np.max(positions_array, axis=0)
        mol_center = (mol_min + mol_max) * 0.5
        mol_extent = np.linalg.norm(mol_max - mol_min)

        # Position camera at a distance that shows all molecules
        # Use molecular extent rather than box size
        camera_distance = max(mol_extent * 2.0, max_radius * 50.0)  # At least 50 molecular radii away

        assert base.cam is not None
        base.cam.setPos(float(mol_center[0]), float(mol_center[1]) - camera_distance, float(mol_center[2]))
        base.cam.lookAt(float(mol_center[0]), float(mol_center[1]), float(mol_center[2]))
    else:
        # Fallback if no molecules placed
        scene_min = np.minimum.reduce(object_state.box_bottom.T)
        scene_max = np.maximum.reduce(object_state.box_top.T)
        center = (scene_min + scene_max) * 0.5
        assert base.cam is not None
        base.cam.setPos(float(center[0]), float(center[1]) - box_diag * 2.0, float(center[2]))
        base.cam.lookAt(float(center[0]), float(center[1]), float(center[2]))

    render_object_state(
        base=base,
        parent=base.render,
        object_state=object_state,
    )

    base.run()


if __name__ == "__main__":
    run_arrangement()
