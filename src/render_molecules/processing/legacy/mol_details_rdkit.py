"""
./src/render_molecules/mol_details_rdkit.py

python -m src.render_molecules.mol_details_rdkit

Uses rdkit to get coordinates, bond list, and other details of a molecule's 3D structure from
SMILES strings.
"""

from rdkit import Chem
from rdkit.Chem import AllChem

from src.utils.json_io import load_json, save_json
from src.utils.type_annotations import (
    Aggregations,
    AtomDetails,
    BondDetails,
    SimDetails,
)


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


def main():
    aggregations: Aggregations = load_json("aggregated.json")
    final = build_details(aggregations=aggregations)
    save_json(final, "final_aggregated.json")


if __name__ == "__main__":
    main()
