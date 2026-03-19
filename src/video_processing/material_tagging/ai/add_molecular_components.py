"""
./src/video_processing/material_tagging/ai/add_molecular_components.py

Takes the annotations.json file, reads the materials for each object, installs a local Ollama model,
adds the molecular composition for each material to the same json, then removes the Ollama model.
"""

import json

import ollama

FOLDER_PATH = "data/vision_json/"
MODEL = "qwen2.5:7b"
COMPOSITION_PROMPT = """
You are an expert chemical engineer. You will receive a JSON object describing a single item in a
scene, including its material composition. Your task is to identify the specific molecules or
compounds that make up those materials and return their chemical formulas.

Rules:
- Output ONLY a JSON object in exactly the format shown below. No explanation, no backticks,
  no extra keys, no wrapping object.
- Keys are molecule or compound names (e.g. "cellulose", "polypropylene"), not material names.
- Values are plain molecular formulas (e.g. "C6H10O5", "C3H6").
- For polymers, give the repeat unit formula only. Never append n, (n), or subscript notation.
- If a material is ambiguous (e.g. "plastic", "metal"), infer the most common specific compound
  for that object type and use that.
- If a material has multiple distinct compounds, list each one as a separate key.
- Do not include duplicate formulas under different keys.

Incorrect output (material names as keys, duplicate formulas):
{
    "composition": {
        "paper": "C6H10O5",
        "wood": "C6H10O5"
    }
}

Correct output (compound names as keys, no duplicates):
{
    "composition": {
        "cellulose": "C6H10O5",
        "lignin": "C9H10O2"
    }
}

Output format (and nothing else):
{
    "composition": {
        "compound_name": "formula",
        "compound_name": "formula"
    }
}

Input:

"""


def load_annotations() -> dict:
    """
    Loads the annotations.

    Returns:
        dict: The string containing the full JSON of the annotations.
    """
    with open(FOLDER_PATH + "annotations.json", "rb") as f:
        annotations: dict = json.load(f)
    return annotations


def use_ollama(object_details: str) -> dict:
    """
    Runs the ollama model given the details of some object inside the annotations JSON.

    Args:
        object_details (str): The details of the object.

    Returns:
        dict: The JSON output from Ollama, being only the composition.
    """
    print(f"Pulling {MODEL} from ollama")
    ollama.pull(MODEL)
    print("Generating responses...")
    response = ollama.generate(
        model=MODEL,
        prompt=COMPOSITION_PROMPT + str(object_details),
        format="json",
    )  # format=json makes valid json output
    # ollama.delete("deepseek-r1:8b")
    return json.loads(response.response)


def aggregate_compositions():
    annotations = load_annotations()
    for obj_name, obj_details in annotations.items():
        response = use_ollama(object_details=obj_details)
        print(response)
        annotations[obj_name]["composition"] = response["composition"]

    return annotations

    # print(annotations)


def save_to_json(final_annotations: dict) -> None:
    with open(FOLDER_PATH + "aggregated.json", "w") as f:
        json.dump(final_annotations, f, indent=2)


aggregated = aggregate_compositions()
save_to_json(aggregated)
