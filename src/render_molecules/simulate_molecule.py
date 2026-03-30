"""
./src/render_molecules/simulate_molecule.py

python -m src.render_molecules.simulate_molecule

Simulates a single object from final_aggregated.json in a Panda3D window.
Picks the first molecule that has sim_details and renders it with CPK colours.
"""

import json
import sys

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
    base: ShowBase, parent: NodePath, sim_details: SimDetails, scale: float = 1.0
) -> NodePath:
    root = parent.attachNewNode("molecule")

    atoms_data = sim_details["atoms"]
    coords_data = sim_details["coords"][0]
    conformer = coords_data["conformers"][0]
    aid_list = coords_data["aid"]  # There are multiple aids

    pos: dict[int, tuple[float, float, float]] = {
        aid: (conformer["x"][i], conformer["y"][i], conformer["z"][i])
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


class MoleculeViewer(ShowBase):
    def __init__(self) -> None:
        super().__init__()
        self.cam.setPos(0, -5, 0)
        self.cam.lookAt(0, 0, 0)

        data = load_json(FINAL_AGGREGATED)

        for obj in data.values():
            for mol in obj.get("composition", {}).values():
                if "sim_details" in mol:
                    build_molecule(self, self.render, mol["sim_details"], scale=3.0)
                    return


if __name__ == "__main__":
    MoleculeViewer().run()
