"""
./src/utils/json_io.py

Generic JSON load/save helpers. All paths are relative to the project root.
"""

import json

FOLDER_PATH = "data/vision_json/"


def load_json(filename: str) -> dict:
    """
    Loads a JSON file from the vision_json folder.

    Args:
        filename (str): Filename to append to FOLDER_PATH (e.g. "annotations.json").

    Returns:
        dict: The parsed JSON content.
    """
    with open(FOLDER_PATH + filename, "rb") as f:
        return json.load(f)


def save_json(data: dict, filename: str) -> None:
    """
    Saves a dict as JSON to the vision_json folder.

    Args:
        data (dict): The data to write.
        filename (str): Filename to append to FOLDER_PATH (e.g. "aggregated.json").
    """
    with open(FOLDER_PATH + filename, "w") as f:
        json.dump(data, f, indent=2)
