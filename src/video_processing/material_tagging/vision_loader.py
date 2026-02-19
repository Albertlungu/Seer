"""
./src/video_processing/material_tagging/vision_loader.py

Loads the Qwen2.5-VL 7B model for later use.
"""

import base64
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
This is an image of a virtual environment. You are to very briefly (in as little words as possible)
label each of the objects in the photo with the following features, in JSON format. In parentheses
are the labels to put to each category:
- item name (name)
- material or materials it is most likely made of, being incredibly precise with the specific type
(e.g. instead of vague fabric, something more precise, such as cotton) (material)

Do not include triple backticks in your response.

EXAMPLE:

json
{
  "bed": {
    "name": "bed",
    "material": "synthetic leather"
  },
  "pillow": {
    "name": "pillow",
    "material": "polyester"
  },
  "duvet": {
    "name": "duvet",
    "material": "cotton"
  },
  "nightstand": {
    "name": "nightstand",
    "material": "acrylic glass and MDF"
  },
  "drawer": {
    "name": "drawer",
    "material": "MDF with metal handles"
  },
  "shelf": {
    "name": "shelf",
    "material": "wood composite"
  },
  "photo_frames": {
    "name": "photo frames",
    "material": "wood and glass"
  },
  "spray_can": {
    "name": "spray can",
    "material": "aluminum"
  },
  "remote_control": {
    "name": "remote control",
    "material": "plastic"
  },
  "floor": {
    "name": "floor",
    "material": "laminate wood"
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
            return response
        except TypeError as e:
            return f"There was an error while generating a response: \n\n {e}"


vision = VisionInstance(vision_api_key, "data/env_imgs/albert_room/frame_0001.jpg")
print(vision.make_completion())
