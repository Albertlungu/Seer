"""
./src/video_processing/material_tagging/ai/grounding_loader.py

Using the Grounding DINO object detection model to calculate bounding boxes.
"""

import json
from typing import Any, Literal, TypedDict, cast

import torch
from colorama import Fore, init
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from ..raycasting import Detections

init(autoreset=True)

IMAGES_PATH = "./data/env_imgs/albert_room"
JSON_PATH = "./data/vision_json/albert_room.json"


class Result(TypedDict):
    boxes: torch.Tensor  # Shape: (N, 4), where N is number of objects
    scores: torch.Tensor  # Shape: (N,) the number of objects
    labels: list[str]


def setup_torch() -> tuple[Literal["cuda", "mps"], str, Any, Any]:
    """
    Sets up the Grounding DINO model with PyTorch

    Returns:
        tuple: Contains the processor, model, device, and model id.
    """

    device = "cuda" if torch.cuda.is_available() else "mps"
    model_id = "IDEA-Research/grounding-dino-tiny"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    return device, model_id, processor, model


def run_detection(
    image_folder: str,
    all_detections: Detections,
    setup_output: tuple[Literal["cuda", "mps"], str, Any, Any],
    debug: bool = False,
) -> dict[str, Result]:
    """
    Runs the Grounding DINO model on a folder of images to return the bboxes.

    Args:
        image_folder (str): The path to the folder containing frames.
        all_detections (dict): Contains the Gemma-made object classification.
        debug (bool, optional): Whether to print debug messages. Defaults to False.

    Returns:
        dict[str, Result]: A dictionary referring one result to each frame.
                            Result - dict:
                                - scores: List of shape [N] with confidence values 0-1
                                    Each entry refers to a single object's confidence value
                                - boxes: List of shape [N, 4] in pixel XYXY format
                                    [x_min, y_min, x_max, y_max], where N is number of objects
                                    Each entry is the bounding box of a single object
                                - labels: List of N strings
                                    Each string is a text label for each box, i.e. an object.
    """
    device: Literal["cuda", "mps"] = setup_output[0]
    processor: Any = setup_output[2]
    model: Any = setup_output[3]

    results: dict[str, Result] = {}

    for frame_name, obj in all_detections.items():
        frame_path = image_folder + f"/{frame_name}"

        if debug:
            print(f"{Fore.RED} DEBUG: {frame_name}")
            print(f"{Fore.RED} DEBUG: {frame_path}")

        image = Image.open(frame_path)

        text = ""  # initialize as an empty string

        for obj_name, obj_data in obj.items():
            text += f" . {obj_name}" if text else obj_name
            # If it is the first element, do not include " . "
        if debug:
            print(f"{Fore.RED} DEBUG: {text}")

        inputs = processor(images=image, text=text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results[frame_name] = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            # box_threshold=0.35,
            text_threshold=0.25,
            target_sizes=[
                image.size[::-1]
            ],  # reversing the values: sequence[start:stop:increment], start & stop are default, increment -1
        )[0]  # is originally a list of a single dict
        if debug:
            print(f"{Fore.BLUE} DEBUG: {results[frame_name]}")

        for key, value in results[frame_name].items():
            if not isinstance(value, list):
                tensor_value = cast(
                    torch.Tensor, value
                )  # `cast` says "this is a Tensor, trust me"
                results[frame_name][key] = (
                    tensor_value.tolist()
                )  # so there's no linting errors

        if debug:
            print(f"{Fore.GREEN} DEBUG: {results[frame_name]}")

    return results


def save_results(output_folder: str, results: dict[str, Result]) -> None:
    with open(output_folder + "/grounding.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    with open(JSON_PATH, "r") as f:
        detections: Detections = json.load(f)

    results = run_detection(IMAGES_PATH, detections, setup_torch(), debug=True)
    save_results(output_folder="./data/vision_json", results=results)


if __name__ == "__main__":
    main()
