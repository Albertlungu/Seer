"""
./src/video_processing/material_tagging/vision_loader.py

Loads the Qwen2.5-VL 7B model for later use.
"""

import base64
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

vision_api_key = os.getenv("VISION_API_KEY")


class VisionInstance:
    def __init__(self, api_key: str | None, image_path: str) -> None:
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.api_key = api_key
        self.image_path = image_path

    @staticmethod
    def image_to_data_url(path: str) -> str:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def make_completion(self):
        url = self.image_to_data_url(self.image_path)

        completion = self.client.chat.completions.create(
            extra_headers={},
            extra_body={},
            model="nvidia/nemotron-nano-12b-v2-vl:free",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": url},
                        },
                        {
                            "type": "text",
                            "text": """
Analyze this image and identify every visible object.

For each object, return:
- "name": the object name (lowercase, concise)
- "materials": an array of the specific materials it is made of

Rules:
1. Be precise with materials. Use specific types, not generic ones:
   - Instead of "fabric" -> "cotton", "polyester", "nylon", "linen"
   - Instead of "plastic" -> "ABS", "polycarbonate", "polypropylene", "PVC"
   - Instead of "metal" -> "aluminum", "steel", "brass", "iron"
   - Instead of "wood" -> "oak", "pine", "MDF", "plywood", "particleboard"
2. If an object has multiple distinct materials, list each one separately in the array.
3. Only output raw JSON. No markdown, no backticks, no explanation.
4. Use snake_case for object keys.

Required output format (and nothing else):

{
  "bed_frame": {
    "name": "bed frame",
    "materials": ["pine", "steel"]
  },
  "pillow": {
    "name": "pillow",
    "materials": ["cotton", "polyester"]
  }
}
""",
                        },
                    ],
                }
            ],
        )

        try:
            response = completion.choices[0].message.content
            if response is None:
                return "Error: model returned no content."
            parsed = json.loads(response)
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            return response  # type: ignore[return-value]
        except TypeError as e:
            return f"There was an error while generating a response: \n\n {e}"


vision = VisionInstance(vision_api_key, "data/env_imgs/albert_room/frame_0001.jpg")
print(vision.make_completion())
