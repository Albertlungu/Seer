"""
./src/utils/json_io.py

Generic JSON load/save helpers. All paths are relative to the project root.
"""

import json

from src.utils.resource_path import resource_path

_FOLDER = "data/vision_json/"


def load_json(filename: str) -> dict:
    """
    Loads a JSON file from the vision_json folder.

    Args:
        filename (str): Filename relative to vision_json/ (e.g. "annotations.json").

    Returns:
        dict: The parsed JSON content.
    """
    with open(resource_path(_FOLDER + filename), "rb") as f:
        return json.load(f)


def save_json(data: dict, filename: str) -> None:
    """
    Saves a dict as JSON to the vision_json folder.

    Args:
        data (dict): The data to write.
        filename (str): Filename relative to vision_json/.
    """
    with open(resource_path(_FOLDER + filename), "w") as f:
        json.dump(data, f, indent=2)
