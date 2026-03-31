"""
./src/render_molecules/simulate_molecule.py

python -m src.render_molecules.simulate_molecule

Simulates a single object from final_aggregated.json in a Panda3D window.
Picks the first molecule that has sim_details and renders it with CPK colours.
"aid"s are the Atom IDs. In the bonds section, "aid1" means starting atom ID, and "aid2" end atom ID

"""

import json
import sys

import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import NodePath, Point3

from src.utils.json_io import load_json
from src.utils.type_annotations import SimDetails

FINAL_AGGREGATED = "final_aggregated.json"

ELEMENT_COLORS: dict[int, tuple[float, float, float]] = {
    1: (1.0, 1.0, 1.0),  # Hydrogen
    6: (0.3, 0.3, 0.3),  # Carbon
    7: (0.2, 0.4, 1.0),  # Nitrogen
    8: (1.0, 0.2, 0.2),  # Oxygen
    14: (0.7, 0.5, 0.1),  # Silicon
    16: (1.0, 0.9, 0.0),  # Sulfur
    20: (0.5, 0.5, 0.5),  # Calcium
    26: (0.6, 0.3, 0.1),  # Iron
}

ELEMENT_RADII: dict[int, float] = {
    1: 0.05,
    6: 0.08,
    7: 0.08,
    8: 0.08,
    14: 0.10,
    16: 0.10,
    20: 0.12,
    26: 0.12,
}

DEFAULT_COLOR: tuple[float, float, float] = (0.8, 0.0, 0.8)
DEFAULT_RADIUS: float = 0.08


def draw_atom(
    base: ShowBase,
    parent: NodePath,
    x: float,
    y: float,
    z: float,
    element: int,
    scale: float = 1.0,
) -> NodePath:
    color = ELEMENT_COLORS.get(element, DEFAULT_COLOR)
    radius = ELEMENT_RADII.get(element, DEFAULT_RADIUS) * scale
    sphere = base.loader.loadModel("models/misc/sphere")
    sphere.reparentTo(parent)
    sphere.setPos(x, y, z)
    sphere.setScale(radius)
    sphere.setColor(*color, 1.0)
    return sphere


def draw_bond(
    base: ShowBase,
    parent: NodePath,
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    order: int | float = 1,
) -> NodePath:
    a = Point3(*p1)
    b = Point3(*p2)
    mid = (a + b) / 2.0
    length = (b - a).length()

    bond = base.loader.loadModel("models/misc/sphere")
    bond.reparentTo(parent)
    bond.setPos(mid)
    bond.setScale(0.02, 0.02, length / 2.0)
    bond.lookAt(Point3(*p2))
    bond.setHpr(bond.getH(), bond.getP() - 90, bond.getR())

    grey = 0.6 - 0.1 * (order - 1)
    bond.setColor(grey, grey, grey, 1.0)
    return bond


def build_molecule(
    base: ShowBase,
    parent: NodePath,
    sim_details: SimDetails,
    scale: float = 1.0,
    offset_axis: str = "x",
    offset_value: float = 0.0,
) -> NodePath:
    root = parent.attachNewNode("molecule")

    atoms_data = sim_details["atoms"]
    coords_data = sim_details["coords"][0]
    conformer = coords_data["conformers"][0]
    aid_list = coords_data["aid"]

    if offset_axis == "x":
        ox, oy, oz = offset_value, 0.0, 0.0
    elif offset_axis == "y":
        ox, oy, oz = 0.0, offset_value, 0.0
    elif offset_axis == "z":
        ox, oy, oz = 0.0, 0.0, offset_value
    else:
        raise ValueError("offset_axis must be one of: 'x', 'y', 'z'")

    pos: dict[int, tuple[float, float, float]] = {
        aid: (conformer["x"][i] + ox, conformer["y"][i] + oy, conformer["z"][i] + oz)
        for i, aid in enumerate(aid_list)
    }
    elem: dict[int, int] = {
        aid: atoms_data["element"][i] for i, aid in enumerate(atoms_data["aid"])
    }

    for aid, (x, y, z) in pos.items():
        draw_atom(base, root, x, y, z, elem.get(aid, 6), scale=scale)

    if "bonds" in sim_details:
        bonds = sim_details["bonds"]
        for a1, a2, order in zip(bonds["aid1"], bonds["aid2"], bonds["order"]):
            if a1 in pos and a2 in pos:
                draw_bond(base, root, pos[a1], pos[a2], order)

    return root


def build_multiple_molecules(
    base: ShowBase,
    parent: NodePath,
    sim_details: SimDetails,
    scale: float = 1.0,
    start_offset: float = 0.0,
    num_molecules: int = 21,
):
    atoms_data = sim_details["atoms"]
    bonds_data = sim_details["bonds"]
    coords_data = sim_details["coords"]

    conformer: dict[str, list[float]] = coords_data[0]["conformers"][0]
    aids: list[int] = coords_data[0]["aid"]
    x_list, y_list, z_list = conformer["x"], conformer["y"], conformer["z"]

    # Find atom closest to origin using d = \sqrt{x^2 + y^2 + z^2}
    aids_np = np.asarray(aids, dtype=int)
    coords_np = np.column_stack((x_list, y_list, z_list))

    # Use squared distance for argmin (same result as sqrt, less work)
    dist2 = np.einsum("ij,ij->i", coords_np, coords_np)
    origin_atom_idx = int(np.argmin(dist2))
    origin_atom_id = int(aids_np[origin_atom_idx])
    origin_atom_coords = (
        float(coords_np[origin_atom_idx, 0]),
        float(coords_np[origin_atom_idx, 1]),
        float(coords_np[origin_atom_idx, 2]),
    )

    # Find atom farthest from origin atom on all axes
    deltas_np = coords_np - coords_np[origin_atom_idx]
    farthest_idx_by_axis = np.argmax(deltas_np, axis=0)

    delta_x = float(
        deltas_np[farthest_idx_by_axis[0], 0]
    )  # Delta x is mainly what I need for the offset, I could care less about the atom ID
    # The only reason I would care about atom ID is for bonding and IMFs that are present
    delta_y = float(deltas_np[farthest_idx_by_axis[1], 1])
    delta_z = float(deltas_np[farthest_idx_by_axis[2], 2])

    farthest_x_atom_idx = int(farthest_idx_by_axis[0])  # Used for bonding/IMF
    farthest_y_atom_idx = int(farthest_idx_by_axis[1])
    farthest_z_atom_idx = int(farthest_idx_by_axis[2])

    farthest_x_atom_id = int(aids_np[farthest_x_atom_idx])
    farthest_y_atom_id = int(aids_np[farthest_y_atom_idx])
    farthest_z_atom_id = int(aids_np[farthest_z_atom_idx])

    # Build one molecule in each direction per iteration, each using its own axis delta.
    current_offset_x = start_offset
    current_offset_y = start_offset
    current_offset_z = start_offset
    for i in range(num_molecules // 3):
        current_offset_x += delta_x
        current_offset_y += delta_y
        current_offset_z += delta_z

        build_molecule(
            base=base,
            parent=parent,
            sim_details=sim_details,
            scale=scale,
            offset_axis="x",
            offset_value=current_offset_x,
        )
        build_molecule(
            base=base,
            parent=parent,
            sim_details=sim_details,
            scale=scale,
            offset_axis="y",
            offset_value=current_offset_y,
        )
        build_molecule(
            base=base,
            parent=parent,
            sim_details=sim_details,
            scale=scale,
            offset_axis="z",
            offset_value=current_offset_z,
        )


# """
# What I wanna do to put multiple molecules next to each other with accurate offset:
# - Go through all atoms in the molecule, select atom farthest from origin on each axis (max(pos.x/y/z))
# - Get position of atom closest to origin.
# - Calculate ∆pos between these two atoms on each axis (as a vector)
# - Add it to the offset.
# """


class MoleculeViewer(ShowBase):
    def __init__(self) -> None:
        super().__init__()
        self.cam.setPos(0, -5, 0)
        self.cam.lookAt(0, 0, 0)

        data = load_json(FINAL_AGGREGATED)
        start_offset = 0.0

        for obj in data.values():
            for mol in obj.get("composition", {}).values():
                if "sim_details" in mol:
                    build_multiple_molecules(
                        self,
                        self.render,
                        mol["sim_details"],
                        scale=3.0,
                        start_offset=start_offset,
                        num_molecules=20,
                    )
                    return


if __name__ == "__main__":
    MoleculeViewer().run()
