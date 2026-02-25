"""
./src/video_processing/material_tagging/gemma_loader.py

Loads the Gemma 3 model from google-genai
"""

import json
import os
from pathlib import Path

import PIL.Image
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

IMAGES_PATH = "./data/env_imgs/albert_room"
OUTPUT_FOLDER = "./data/vision_json"


class Gemma:
    def __init__(self, image_folder: str, output_folder: str, debug: bool = False) -> None:
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = "gemma-3-27b-it"
        self.image_folder = image_folder
        self.output_folder = output_folder
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
            prompt = base_prompt + """
The following is the output from the last frame that you predicted. You must attempt to be as
consistent as possible, but keep in mind that new items may have appeared.

""" + prev_resp
        else:
            prompt = base_prompt

        img = PIL.Image.open(image_path)
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

    def run_detection(self, n: int = 10) -> dict:
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
                resp = self.get_response(image_path=image_path, prev_resp=str(prev_resp))

            results[frame_name] = resp
            prev_resp = resp

            if self.debug:
                print("DEBUG:", resp)

        return results

    def save_results(self, results: dict) -> None:
        """
        Saves results to a JSON file.

        Args:
            results (dict): The results to save.
        """
        with open(self.output_folder + "/gemma.json", "w") as f:
            json.dump(results, f, indent=2)


gemma = Gemma(IMAGES_PATH, OUTPUT_FOLDER, debug=False)
results = gemma.run_detection(n=10)
gemma.save_results(results)
