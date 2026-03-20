"""
./src/video_processing/material_tagging/ai/add_molecular_components.py

Takes the annotations.json file, reads the materials for each object, installs a local Ollama model,
adds the molecular composition for each material to the same json, then removes the Ollama model.

Uses the molecule names in the aggregated JSON file to lookup SMILES.
"""

import json

import ollama
import requests

from src.utils.type_annotations import (
    AnnotatedObjectDetails,
    Annotations,
    Aggregations,
)

FOLDER_PATH = "data/vision_json/"
MODEL = "qwen2.5:7b"
COMPOSITION_PROMPT = """
You are an expert chemical engineer. You will receive a JSON object describing a single item in a
scene, including its material composition. Your task is to identify the specific molecules or
compounds that make up those materials and return their chemical formulas.

Rules:
- Output ONLY a JSON object in exactly the format shown below. No explanation, no backticks,
  no extra keys, no wrapping object.
- Keys are the names of specific small molecules or discrete compounds that are queryable in
  PubChem by name. For polymers, use the repeat unit or monomer name, not the polymer name.
  Examples: use "cellobiose" not "cellulose", use "coniferyl-alcohol" not "lignin",
  use "propylene" not "polypropylene", use "ethylene" not "polyethylene",
  use "glycyl-prolyl-hydroxyproline" not "collagen".
- Each compound key maps to a dict with a single key "formula" whose value is the plain molecular
  formula. Never append n, (n), or subscript notation.
- Keys must NOT be material names (e.g. "wood", "paper", "metal", "leather", "plastic").
- If a material is ambiguous (e.g. "plastic", "metal"), infer the most common specific compound
  for that object type and use that.
- If a material has multiple distinct compounds, list each one as a separate key.
- Do not include duplicate formulas under different keys.

Incorrect output (polymer names and material names as keys, flat string values):
{
    "composition": {
        "cellulose": "C6H10O5",
        "wood": "C6H10O5"
    }
}

Correct output (PubChem-queryable small molecule names as keys, dict values):
{
    "composition": {
        "cellobiose": {"formula": "C12H22O11"},
        "coniferyl-alcohol": {"formula": "C10H12O3"}
    }
}

Output format (and nothing else):
{
    "composition": {
        "compound_name": {"formula": "formula"},
        "compound_name": {"formula": "formula"}
    }
}

Input:

"""


def load_annotations() -> Annotations:
    """
    Loads the annotations.

    Returns:
        Annotations: The annotation data keyed by object name.
    """
    with open(FOLDER_PATH + "annotations.json", "rb") as f:
        annotations: Annotations = json.load(f)
    return annotations


def run_ollama(
    object_details: AnnotatedObjectDetails, delete_model: bool = False
) -> dict:
    """
    Runs the ollama model given the details of some object inside the annotations JSON.

    Args:
        object_details (str): The details of the object.

    Returns:
        dict: The JSON output from Ollama, being only the composition.
    """
    print("Generating response...")
    response = ollama.generate(
        model=MODEL,
        prompt=COMPOSITION_PROMPT + str(object_details),
        format="json",
    )  # format=json makes valid json output

    if delete_model:
        ollama.delete(MODEL)

    return json.loads(response.response)


def build_smiles(composition: dict[str, dict[str, str]]) -> None:
    """
    Adds the SMILES string for each molecule name.

    Args:
        composition (dict[str, dict[str, str]]): The composition value taken from the response dictionary.
    """
    for molec_name, molec_details in composition.items():
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{molec_name}/property/IsomericSMILES/JSON"
        response = requests.get(url)
        data = response.json()
        try:
            smiles = data["PropertyTable"]["Properties"][0]["SMILES"]
            molec_details["smiles"] = smiles
        except KeyError as e:
            print(f"Key error: {e}")


def aggregate_compositions() -> Aggregations:
    annotations = load_annotations()
    aggregations: Aggregations = {}
    for obj_name, obj_details in annotations.items():
        response = run_ollama(object_details=obj_details, delete_model=True)
        print(response)
        composition = response["composition"]
        build_smiles(composition)
        aggregations[obj_name] = {**obj_details, "composition": composition}
    return aggregations


def save_to_json(aggregations: Aggregations) -> None:
    """
    Saves the final aggregations to a file.

    Args:
        aggregations (Aggregations): The aggregated dictionary with the compositions.
    """
    with open(FOLDER_PATH + "aggregated.json", "w") as f:
        json.dump(aggregations, f, indent=2)


def main():
    print(f"Pulling {MODEL} from ollama")
    ollama.pull(MODEL)

    aggregated = aggregate_compositions()
    save_to_json(aggregated)
