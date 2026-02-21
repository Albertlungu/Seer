"""
./src/video_processing/material_tagging/vision_loader.py
"""

import base64
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

vision_api_key = os.getenv("VISION_API_KEY")


class VisionInstance:
    def __init__(self, api_key: str | None, image_path: str, output_file: str) -> None:
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.api_key = api_key
        self.image_path = image_path
        self.output_file = output_file

    @staticmethod
    def image_to_data_url(path: str) -> str:
        """
        Turns an image to a data URL (needed for OpenRouter image interpretation)

        Args:
            path (str): Relative path to image

        Returns:
            str: The data URL
        """
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def make_completion(self, image_path: str) -> str:
        """
        Makes a single response from OpenRouter.

        Args:
            image_path (str): Path to image.

        Returns:
            str: The JSON dump of the model's response or error if one is generated.
        """
        url = self.image_to_data_url(image_path)

        completion = self.client.chat.completions.create(
            extra_headers={},
            extra_body={},
            model="meta-llama/llama-3.2-11b-vision-instruct:free",
            max_tokens=4096,
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
            # Strip markdown code fences the model sometimes adds despite instructions
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]  # Drop the opening fence line
                cleaned = cleaned.rsplit("```", 1)[0]  # Drop the closing fence
            parsed = json.loads(cleaned)
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            return response  # type: ignore[return-value]
        except TypeError as e:
            return f"There was an error while generating a response: \n\n {e}"

    def run_nth_frame(self, n: int) -> str:
        """
        Runs the VLM on every Nth frame.

        Args:
            n (int): After how many frames to run the VLM

        Returns:
            str: The final json output
        """
        image_path = list(self.image_path)

        folder_path = Path(image_path[self.image_path.rfind("/")])
        file_count = sum(1 for item in folder_path.iterdir() if item.is_file())

        complete_json = []

        for i in range(file_count):
            image_path[-8:] = f"{i + 1:04}.jpg"
            image_path_str = "".join(image_path)
            temp = self.make_completion(image_path=image_path_str)
            complete_json.append(temp)
            i += n

        complete_json = ",".join(complete_json)

        return complete_json

    def save_to_json(self, json_input: str) -> None:
        """
        Given a JSON-formatted input, saves to an output file

        Args:
            json_input (str): JSON-formatted input.
        """
        with open(self.output_file, "w") as f:
            json.dump(json_input, f)


vision = VisionInstance(
    vision_api_key,
    "data/env_imgs/albert_room/frame_0015.jpg",
    "data/vision_json/albert_room.json",
)
# vlm_output = vision.run_nth_frame(200)
# vision.save_to_json(vlm_output)
temp = vision.make_completion("data/env_imgs/albert_room/frame_0015.jpg")
print(temp)
with open("data/vision_json/test.txt", "w") as f:
    f.write(temp)
