"""
./src/video_processing/material_tagging/qwen_loader.py

Loads the Qwen2.5-VL 7B model for later use.
"""

import base64
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

qwen_api_key = os.getenv("QWEN_API_KEY")


class QwenInstance:
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
            model="qwen/qwen-2.5-vl-72b-instruct",
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
- material it is most likely made of (material)
""",
                        },
                    ],
                }
            ],
        )
        return completion.choices[0].message.content


qwen = QwenInstance(qwen_api_key, "data/env_imgs/albert_room/frame_0001.jpg")
print(qwen.make_completion())
