"""
./src/render_molecules/get_molecule_details.py

Uses rdkit to get coordinates, bond list, and other details of a molecule's 3D structure from
SMILES strings.
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import json
from typing import TypedDict

FOLDER_PATH = "data/vision_json/"


class AggregatedObjectDetails(TypedDict):
    name: str
    materials: list[str]
    corners: dict[str, list[list[float]]]
    base_normal: list[float]
    composition: dict[str, dict]

Aggregations = dict[str, AggregatedObjectDetails]


def load_aggregations() -> dict:
    """
    Loads the aggregations.

    Returns:
        dict: The string containing the full JSON of the aggregations.
    """
    with open(FOLDER_PATH + "aggregations.json", "rb") as f:
        aggregations: dict = json.load(f)
    return aggregations

def build_details(aggregations: Aggregations):

