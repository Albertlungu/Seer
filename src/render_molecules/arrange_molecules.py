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

import numpy as np

from src.render_molecules.arrangement.scene_state import (
    Environment,
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


def create_env():
    data = load_json(FINAL_AGGREGATED)

    environment = Environment()
    environment.scene_states = []

    for object_key, object_details in data.items():
        obj = ObjectState()
        obj.object_key = object_key
        obj.object_name = object_details.get("name", object_key)
        obj.instance_id = object_key
        obj.display_name = object_key.replace("_", " ").title()
        obj.templates = []
        for molecule_name, molecule_details in object_details["composition"].items():
            sim_details: SimDetails | None = molecule_details.get("sim_details")
            if sim_details is None:
                raise ValueError(
                    f"Missing sim_details for molecule '{molecule_name}' in object '{object_key}'"
                )

            atom_details: AtomDetails | None = sim_details.get("atoms")
            bond_details: BondDetails | None = sim_details.get("bonds")
            coords = sim_details.get("coords")
            if not atom_details or not bond_details or not coords:
                raise ValueError(
                    f"Empty sim_details for molecule '{molecule_name}' in object '{object_key}'"
                )

            coords_details: CoordsDetails = coords[0]

            molecule_template = MoleculeTemplate()
            molecule_template.name = molecule_name
            molecule_template.aids = np.array(atom_details["aid"])
            molecule_template.elements = np.array(atom_details["element"])
            molecule_template.local_xyz = (
                np.array(coords_details["conformers"][0]["x"]),
                np.array(coords_details["conformers"][0]["y"]),
                np.array(coords_details["conformers"][0]["z"]),
            )
            molecule_template.bonds_aid1 = np.array(bond_details["aid1"])
            molecule_template.bonds_aid2 = np.array(bond_details["aid2"])
            molecule_template.bond_order = np.array(bond_details["order"])

            obj.templates.append(molecule_template)

        obj.box_bottom = np.array(object_details["corners"]["bottom"])
        obj.box_top = np.array(object_details["corners"]["top"])
        obj.rng_seed = 1

        environment.scene_states.append(obj)
    return environment


print(create_env().scene_states)
