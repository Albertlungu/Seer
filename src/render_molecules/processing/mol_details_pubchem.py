"""
./src/render_molecules/mol_details_pubchem.py

python -m src.render_molecules.mol_details_pubchem

Gets the details for each molecule using PubChem instead of RDKit.
Writes bond details (order, position, etc.), atom details (position, idx) to a JSON file.
"""

import requests
from urllib.parse import quote

from src.utils.json_io import load_json, save_json
from src.utils.type_annotations import Aggregations


def build_details(aggregations: Aggregations):
    for obj_details in aggregations.values():
        for mol_name, mol_details in obj_details["composition"].items():
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(mol_name.replace('-', ' '))}/JSON?record_type=3d"
            response = requests.get(url)
            data = response.json()
            if "Fault" in data:
                print(f"No 3D conformer for {mol_name}: {data['Fault']['Message']}")
                continue
            compound = data["PC_Compounds"][0]
            mol_details["sim_details"] = {
                "atoms": compound["atoms"],
                "bonds": compound["bonds"],
                "coords": compound["coords"],
            }


def main():
    aggregations: Aggregations = load_json("aggregated.json")
    build_details(aggregations)
    save_json(aggregations, "final_aggregated.json")


if __name__ == "__main__":
    main()
