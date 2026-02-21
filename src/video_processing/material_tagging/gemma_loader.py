"""
./src/video_processing/material_tagging/gemma_loader.py

Loads the Gemma 3 model from google-genai
"""

import os

import PIL.Image
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class Gemma:
    def __init__(self, image_path: str) -> None:
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = "gemma-3-27b-it"
        self.image_path = image_path

    def get_response(self):
        prompt = """
Analyze this image and identify every visible object.

For each object, return:
- "name": the object name (lowercase, concise)
- "materials": an array of the specific materials it is made of
- "bounding_box": approximate bounding box as 4 corner coordinates in clockwise order:
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
        img = PIL.Image.open(self.image_path)
        response = self.client.models.generate_content(
            model=self.model, contents=[prompt, img]
        )
        return response.text


gemma = Gemma("data/env_imgs/albert_room/frame_0001.jpg")
print(gemma.get_response())
