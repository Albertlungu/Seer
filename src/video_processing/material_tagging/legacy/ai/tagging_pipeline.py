"""
./src/video_processing/material_tagging/pipeline.py

The full material tagging and bounding box pipeline, including both Gemma and Grounding DINO models.
"""

import json
import os
from pathlib import Path
from typing import TypedDict, cast

import torch
from colorama import Fore, init
from dotenv import load_dotenv
from google import genai
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

load_dotenv()

init(autoreset=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

IMAGES_FOLDER = "./data/env_imgs/albert_room"
GEMMA_OUTPUT = "./data/vision_json/gemma.json"
GROUNDING_OUTPUT = "./data/vision_json/grounding.json"
DETECTION_OUTPUT = "./data/vision_json/full_detections.json"


class ObjectData(TypedDict):
    name: str
    materials: list[str]
    bounding_box: list[list[float]]


class Result(TypedDict):
    scores: torch.Tensor | list  # Shape: (N,) the number of objects
    boxes: torch.Tensor | list[list]  # Shape: (N, 4), where N is number of objects
    labels: list[str]


Detection = dict[str, dict[str, ObjectData]]


class Gemma:
    def __init__(
        self, image_folder: str, gemma_output: str = GEMMA_OUTPUT, debug: bool = False
    ) -> None:
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = "gemma-3-27b-it"
        self.image_folder = image_folder
        self.gemma_output = gemma_output
        self.debug = debug

    def get_response(self, image_path: str, prev_resp: str = "") -> dict:
        """
        Makes a single response from Gemma.

        Args:
            image_path (str): Path to image.
            prev_resp (str): Previous response for consistency.

        Returns:
            dict: The JSON dump of the model's response.
        """
        base_prompt = """
Analyze this image and identify every visible object.

Rules:
1. Be precise with materials. Use specific types, not generic ones:
   - Instead of "fabric" -> "cotton", "polyester", "nylon", "linen"
   - Instead of "plastic" -> "ABS", "polycarbonate", "polypropylene", "PVC"
   - Instead of "metal" -> "aluminum", "steel", "brass", "iron"
   - Instead of "wood" -> "oak", "pine", "MDF", "plywood", "particleboard"
2. If an object has multiple distinct materials, list each one separately in the array.
3. Only output raw JSON. No markdown, no backticks, no explanation.
4. Use snake_case for object keys.
6. You must NEVER include triple backticks in your response.

Required output format (and nothing else):

{
  "bed_frame": {
    "name": "bed frame",
    "materials": ["pine", "steel"],
  },
  "pillow": {
    "name": "pillow",
    "materials": ["cotton", "polyester"],
  }
}

"""
        if prev_resp:
            prompt = (
                base_prompt
                + """
The following is the output from the last frame that you predicted. You must attempt to be as
consistent as possible, but keep in mind that new items may have appeared.

"""
                + prev_resp
            )
        else:
            prompt = base_prompt

        img = Image.open(image_path)
        response = self.client.models.generate_content(
            model=self.model, contents=[prompt, img]
        )
        if response.text:
            cleaned = response.text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                cleaned = cleaned.rsplit("```", 1)[0]
        parsed = json.loads(cleaned)
        return parsed

    def run_gemma(self, n: int = 10) -> dict:
        """
        Runs Gemma on every Nth frame in the image folder.

        Args:
            n (int): Process every Nth frame. Defaults to 10.

        Returns:
            dict: Detection results keyed by frame name.
        """
        folder_path = Path(self.image_folder)
        file_count = sum(1 for item in folder_path.iterdir() if item.is_file())

        results = {}
        prev_resp = None

        for i in range(1, file_count, n):
            frame_name = f"frame_{i:04}.jpg"
            image_path = str(folder_path / frame_name)

            if self.debug:
                print("DEBUG: Running Gemma on", image_path)

            if prev_resp is None:
                resp = self.get_response(image_path=image_path)
            else:
                resp = self.get_response(
                    image_path=image_path, prev_resp=str(prev_resp)
                )

            results[frame_name] = resp
            prev_resp = resp

            if self.debug:
                print("DEBUG:", resp)

        return results

    def save_gemma(self) -> None:
        """
        Saves results to a JSON file.

        Args:
            results (dict): The results to save.
        """
        results = self.run_gemma()
        with open(self.gemma_output, "w") as f:
            json.dump(results, f, indent=2)


class Grounding(Gemma):
    def __init__(self, grounding_output: str = GROUNDING_OUTPUT):

        super().__init__

        self.grounding_output = grounding_output

        if not Path(self.gemma_output).exists():
            self.save_gemma()

        with open(self.gemma_output, "rb") as f:
            self.gemma_detections: dict[str, Detection] = json.load(f)

    def setup_torch(self):
        """
        Sets up the Grounding DINO model with PyTorch
        """

        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.model_id = "IDEA-Research/grounding-dino-tiny"

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)

    def run_grounding(self) -> dict[str, Result]:
        """
        Runs the Grounding DINO model on a folder of images to return the bboxes.

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
        results: dict[str, Result] = {}

        for frame_name, obj in self.gemma_detections.items():
            frame_path = self.image_folder + f"/{frame_name}"

            if self.debug:
                print(f"{Fore.GREEN} DEBUG: {frame_name}")
                print(f"{Fore.GREEN} DEBUG: {frame_path}")

            image = Image.open(frame_path)

            text = ""  # initialize as an empty string

            for obj_name, obj_data in obj.items():
                text += f" . {obj_data['name']}" if text else obj_name
                # If it is the first element, do not include " . "
            if self.debug:
                print(f"{Fore.GREEN} DEBUG: {text}")

            inputs = self.processor(images=image, text=text, return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            results[frame_name] = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                # box_threshold=0.35,
                text_threshold=0.25,
                target_sizes=[
                    image.size[::-1]
                ],  # reversing the values: sequence[start:stop:increment], start & stop are default, increment -1
            )[0]  # is originally a list of a single dict
            if self.debug:
                print(f"{Fore.GREEN} DEBUG: {results[frame_name]}")

            for key, value in results[frame_name].items():
                if not isinstance(value, list):
                    tensor_value = cast(
                        torch.Tensor, value
                    )  # `cast` says "this is a Tensor, trust me"
                    results[frame_name][key] = (
                        tensor_value.tolist()
                    )  # so there's no linting errors

            if self.debug:
                print(f"{Fore.CYAN} DEBUG: {results[frame_name]}")

        return results

    def normalize(self) -> dict[str, Result]:
        """
        Returns the normalized coordinates for each corner of a bounding box.

        Returns:
            dict[str, Result]: Same results dictionary but with normalized values for bounding boxes, and in corner format.
        """
        image = Image.open(self.image_folder + "/frame_0001.jpg")
        w, h = image.size  # Getting image size in px

        results = self.run_grounding()
        n_results = results

        for frame_name, result in results.items():
            boxes = []  # empty list
            for box in result["boxes"]:
                x_min, y_min, x_max, y_max = box
                nx_min = x_min / w
                ny_min = y_min / h
                nx_max = x_max / w
                ny_max = y_max / h
                corners = [
                    [nx_min, ny_min],  # top left
                    [nx_max, ny_min],  # top right
                    [nx_min, ny_max],  # bottom left
                    [nx_max, ny_max],  # bottom right
                ]
                boxes.append(corners)
            n_results[frame_name]["boxes"] = boxes

        return n_results

    def save_grounding(self) -> None:
        """
        Saves a set of results to a JSON file
        """
        results = self.normalize()
        with open(self.grounding_output, "w") as f:
            json.dump(results, f, indent=2)


def build_detection_json(
    gemma_output: str = GEMMA_OUTPUT, grounding_output: str = GROUNDING_OUTPUT
) -> dict[str, Detection]:
    """
    Builds the full detection dict, putting Gemma and Grounding together.

    Returns:
        dict[str, Detection]: The Gemma dict but in the obj data, there's also the boxes from Grounding.
    """
    if Path(gemma_output).exists():
        with open(gemma_output, "rb") as f:
            frame_detections = json.load(f)

    if Path(grounding_output).exists():
        with open(grounding_output, "rb") as f:
            grounding_detections = json.load(f)

        for frame_name, objects in frame_detections.items():
            g = grounding_detections[frame_name]
            boxes: list = g["boxes"]
            labels: list = g["labels"]
            scores: list = g["scores"]

            # For each Gemma object, find the best matching Grounding DINO detection
            for obj_key, obj_data in objects.items():
                obj_name = obj_data["name"].lower()
                best_box = None
                best_score = -1.0

                for label, box, score in zip(labels, boxes, scores):
                    if label.lower() == obj_name and score > best_score:
                        best_box = box
                        best_score = score

                if best_box is not None:
                    obj_data["bounding_box"] = best_box
                else:
                    obj_data["bounding_box"] = []
    else:
        print(f"{Fore.RED} File did not exist.")

    return frame_detections


def save_detection(
    frame_detections: dict[str, Detection], detection_output: str = DETECTION_OUTPUT
):
    """
    Saves the full detection to a JSON file.
    """
    all_detections = frame_detections
    with open(detection_output, "w") as f:
        json.dump(all_detections, f, indent=2)
    print(f"{Fore.GREEN}Built full Detections JSON. Check {detection_output}")


def main():
    frame_detections = build_detection_json()
    save_detection(frame_detections=frame_detections)


if __name__ == "__main__":
    main()
