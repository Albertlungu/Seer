"""
./src/render_molecules/mol_details_pubchem.py

python -m src.render_molecules.mol_details_pubchem

Gets the details for each molecule using PubChem instead of RDKit.
Writes bond details (order, position, etc.), atom details (position, idx) to a JSON file.
"""

from urllib.parse import quote

import requests

from src.utils.json_io import load_json, save_json
from src.utils.type_annotations import Aggregations


def fetch_pubchem_data(mol_name: str, smiles: str | None) -> dict | None:
    endpoints = [
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(mol_name.replace('-', ' '))}/JSON?record_type=3d",
    ]
    if smiles:
        endpoints.append(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{quote(smiles)}/JSON?record_type=3d"
        )

    endpoints.append(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(mol_name.replace('-', ' '))}/JSON"
    )

    if smiles:
        endpoints.append(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{quote(smiles)}/JSON"
        )

    for url in endpoints:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "Fault" not in data:
                return data
    return None


def build_details(aggregations: Aggregations):
    for obj_details in aggregations.values():
        composition = obj_details["composition"]
        molecules_to_remove: list[str] = []
        for mol_name, mol_details in composition.items():
            data = fetch_pubchem_data(mol_name, mol_details.get("smiles"))

            if not data:
                print(f"Could not fetch data for {mol_name}")
                molecules_to_remove.append(mol_name)
                continue

            compound = data["PC_Compounds"][0]
            atoms = compound.get("atoms")
            bonds = compound.get("bonds", {"aid1": [], "aid2": [], "order": []})
            coords = compound.get("coords")

            # Pad Z dimension if working with 2D fallback data
            if coords and len(coords) > 0 and "conformers" in coords[0]:
                conf = coords[0]["conformers"][0]
                if "x" in conf and "z" not in conf:
                    conf["z"] = [0.0] * len(conf["x"])

            atoms_ok = bool(atoms and atoms.get("aid") and atoms.get("element"))
            bonds_ok = bool(
                bonds is not None
                and "aid1" in bonds
                and "aid2" in bonds
                and "order" in bonds
            )
            coords_ok = bool(coords and len(coords) > 0 and coords[0].get("conformers"))

            if not (atoms_ok and bonds_ok and coords_ok):
                print(f"Skipping {mol_name}: incomplete PubChem data")
                molecules_to_remove.append(mol_name)
                continue

            mol_details["sim_details"] = {
                "atoms": atoms,
                "bonds": bonds,
                "coords": coords,
            }

        for mol_name in molecules_to_remove:
            composition.pop(mol_name, None)


def main():
    aggregations: Aggregations = load_json("aggregated.json")
    build_details(aggregations)
    save_json(aggregations, "final_aggregated.json")


if __name__ == "__main__":
    main()
