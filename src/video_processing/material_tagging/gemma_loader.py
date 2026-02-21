"""
./src/video_processing/material_tagging/gemma_loader.py

Loads the Gemma 3 model from google-genai
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import json

import PIL.Image
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class Gemma:
    def __init__(self, image_path: str, output_file: str) -> None:
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = "gemma-3-27b-it"
        self.image_path = image_path
        self.output_file = output_file

    def get_response(self, image_path) -> dict:
        """
        Makes a single response from OpenRouter.

        Args:
            image_path (str): Path to image.

        Returns:
            dict: The JSON dump of the model's response or error if one is generated.
        """
        prompt = """
Analyze this image and identify every visible object.

For each object, return:
[top_left, top_right, bottom_right, bottom_left]. Each corner is [x, y] where x and y are
normalized from 0.0 to 1.0 relative to image width and height (e.g. [0.5, 0.5] = center of image,
[0.0, 0.0] = top-left corner of image, [1.0, 1.0] = bottom-right corner of image).

Rules:
1. Be precise with materials. Use specific types, not generic ones:
   - Instead of "fabric" -> "cotton", "polyester", "nylon", "linen"
   - Instead of "plastic" -> "ABS", "polycarbonate", "polypropylene", "PVC"
   - Instead of "metal" -> "aluminum", "steel", "brass", "iron"
   - Instead of "wood" -> "oak", "pine", "MDF", "plywood", "particleboard"
2. If an object has multiple distinct materials, list each one separately in the array.
3. Only output raw JSON. No markdown, no backticks, no explanation.
4. Use snake_case for object keys.
5. Bounding boxes should tightly enclose each object. Estimate the position as accurately as
possible based on where the object appears in the image.
6. You must NEVER include triple backticks in your response.

Required output format (and nothing else):

{
  "bed_frame": {
    "name": "bed frame",
    "materials": ["pine", "steel"],
    "bounding_box": [[0.1, 0.3], [0.8, 0.3], [0.8, 0.9], [0.1, 0.9]]
  },
  "pillow": {
    "name": "pillow",
    "materials": ["cotton", "polyester"],
    "bounding_box": [[0.2, 0.25], [0.45, 0.25], [0.45, 0.45], [0.2, 0.45]]
  }
}
"""
        img = PIL.Image.open(image_path)
        response = self.client.models.generate_content(
            model=self.model, contents=[prompt, img]
        )
        if response.text:
            cleaned = response.text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]  # Drop the opening fence line
                cleaned = cleaned.rsplit("```", 1)[0]  # Drop the closing fence
        parsed = json.loads(cleaned)
        return parsed

    def run_nth_frame(self, n: int):
        """
        Runs the VLM on every Nth frame.

        Args:
            n (int): After how many frames to run the VLM

        Returns:
            str: The final json output
        """

        folder_path = Path(self.image_path[: self.image_path.rfind("/")])
        file_count = sum(1 for item in folder_path.iterdir() if item.is_file())

        results = {}
        image_path = list(self.image_path)

        i = 0
        while i < file_count:
            image_path[-8:] = f"{i + 1:04}.jpg"
            image_path_str = "".join(image_path)
            temp = self.get_response(image_path=image_path_str)
            results[i + 1] = temp  # Keyed by frame number
            i += n

        return results

    def save_to_json(self, json_input: dict) -> None:
        """
        Given a JSON-formatted input, saves to an output file

        Args:
            json_input (dict): JSON-formatted input.
        """
        with open(self.output_file, "w") as f:
            json.dump(json_input, f, indent=2)


gemma = Gemma(
    "data/env_imgs/albert_room/frame_0001.jpg", "data/vision_json/albert_room.json"
)
result = gemma.run_nth_frame(10)
gemma.save_to_json(result)
