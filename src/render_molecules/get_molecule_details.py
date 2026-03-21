"""
./src/render_molecules/get_molecule_details.py

python -m src.render_molecules.get_molecule_details

Uses rdkit to get coordinates, bond list, and other details of a molecule's 3D structure from
SMILES strings.
"""

import json

from rdkit import Chem
from rdkit.Chem import AllChem

from src.utils.type_annotations import (
    Aggregations,
    AtomDetails,
    BondDetails,
    SimDetails,
)

FOLDER_PATH = "data/vision_json/"


def load_aggregations() -> Aggregations:
    """
    Loads the aggregations.

    Returns:
        Aggregations: The aggregated data keyed by object name.
    """
    with open(FOLDER_PATH + "aggregated.json", "rb") as f:
        aggregations: Aggregations = json.load(f)
    return aggregations


def get_details(smiles: str) -> SimDetails:
    molecule = Chem.MolFromSmiles(smiles)
    molecule = Chem.AddHs(molecule)
    AllChem.EmbedMolecule(molecule, AllChem.ETKDGv3())  # type: ignore[attr-defined]

    conf = molecule.GetConformer()

    atoms: list[AtomDetails] = [
        {
            "idx": atom.GetIdx(),
            "symbol": atom.GetSymbol(),
            "position": list(conf.GetAtomPosition(atom.GetIdx())),
        }
        for atom in molecule.GetAtoms()
    ]

    bonds: list[BondDetails] = [
        {
            "begin": bond.GetBeginAtomIdx(),
            "end": bond.GetEndAtomIdx(),
            "type": bond.GetBondTypeAsDouble(),
        }
        for bond in molecule.GetBonds()
    ]

    return {"atoms": atoms, "bonds": bonds}


def build_details(aggregations: Aggregations) -> Aggregations:
    for obj_details in aggregations.values():
        composition = obj_details.get("composition", {})
        for molec_details in composition.values():
            smiles = molec_details["smiles"]
            molec_details["sim_details"] = get_details(smiles)
    return aggregations


def save_aggregations(aggregations: Aggregations):
    with open(FOLDER_PATH + "final_aggregated.json", "w") as f:
        json.dump(aggregations, f, indent=2)


def main():
    aggregations = load_aggregations()
    final = build_details(aggregations=aggregations)
    save_aggregations(final)


if __name__ == "__main__":
    main()
