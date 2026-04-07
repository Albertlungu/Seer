"""
./src/render_molecules/arrange_molecules.py

python -m src.render_molecules.arrange_molecules

Arrangement pipeline entrypoint module.

Serves as the top-level orchestration script for building realistic static molecular arrangements from aggregated JSON input.

- Load composition and molecular conformer data from project JSON sources.
- Construct template/state records for arrangement processing.
- Invoke placement and optional static relaxation routines.
- Hand arranged scene-state data to the renderer adapter for Panda3D visualization.

- Replace exploratory simulation script usage with a production-oriented arrangement entrypoint.
- Keep orchestration logic thin and delegate details to state, geometry, placement, and renderer modules.
- Main application class or runner function.
- Configuration loading for box bounds, composition counts, seeds, and packing settings.
- Startup and shutdown wiring for Panda3D scene lifecycle.
"""

import argparse

import numpy as np
from direct.showbase.ShowBase import ShowBase

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
    for key in ("count", "quantity", "target_count", "instances", "copies"):
        value = molecule_details.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return 1


def build_object_state(
    object_key: str | None = None,
) -> tuple[ObjectState, dict[int, int]]:
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
    seed: int, target_counts: dict[int, int]
) -> PlacementConfig:
    target_total = int(sum(target_counts.values()))
    return PlacementConfig(
        seed=seed,
        max_total_attempts=max(1000, target_total * 100),
        frontier_radius=0.35,
        target_instance_count=target_total,
        stop_when_target_met=True,
        require_in_bounds=True,
        require_no_overlap=True,
    )


class ArrangementApp(ShowBase):
    def __init__(self, object_state: ObjectState) -> None:
        super().__init__()
        self.disableMouse()
        self.setBackgroundColor(0.08, 0.08, 0.1, 1.0)
        if self.camera is not None:
            self.camera.setPos(0.0, -8.0, 2.5)
            self.camera.lookAt(0.0, 0.0, 0.0)
        self.rendered_roots = render_object_state(
            base=self,
            parent=self.render,
            object_state=object_state,
        )


def main() -> None:
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
    config = _default_placement_config(seed=args.seed, target_counts=target_counts)
    arranged_state = place_molecules(
        object_state=object_state,
        config=config,
        target_counts=target_counts,
    )

    app = ArrangementApp(object_state=arranged_state)
    app.run()


if __name__ == "__main__":
    main()
