"""
./src/video_processing/view_bbox.py

Views bboxes in 2D from Gemma to check if any misalignments are caused by the 3D conversion or
if it's because of Gemma.
"""

import json
import os

from PIL import Image, ImageDraw, ImageFont

JSON_PATH = "./data/vision_json/full_detections.json"
IMAGES_PATH = "./data/env_imgs/albert_room"
OUTPUT_PATH = "./logs/bboxes"

COLORS = [
    "red",
    "green",
    "blue",
    "yellow",
    "magenta",
    "cyan",
    "orange",
    "purple",
]


def draw_bboxes(
    json_path: str, images_path: str, output_path: str, frame_name: str | None = None
):
    """
    Draws bounding boxes on frames from Gemma's detections.

    Args:
        json_path: Path to the JSON file with detections.
        images_path: Path to the image directory.
        output_path: Path to save annotated images.
        frame_name: If provided, only draw this frame. Otherwise draw all.
    """
    os.makedirs(output_path, exist_ok=True)

    with open(json_path, "r") as f:
        detections = json.load(f)

    frames = {frame_name: detections[frame_name]} if frame_name else detections

    for fname, objects in frames.items():
        img_path = os.path.join(images_path, fname)
        if not os.path.exists(img_path):
            print(f"Skipping {fname} (image not found)")
            continue

        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        w, h = img.size

        for i, (obj_name, obj_data) in enumerate(objects.items()):
            color = COLORS[i % len(COLORS)]
            bbox = obj_data["bounding_box"]
            if not bbox:
                continue

            # Convert normalized coords to pixel coords
            # bbox corners: [TL, TR, BL, BR]
            points = [(nx * w, ny * h) for nx, ny in bbox]

            # Reorder to TL, TR, BR, BL for a proper rectangle polygon
            tl, tr, bl, br = points
            rect = [tl, tr, br, bl]
            draw.polygon(rect, outline=color, width=3)

            # Draw corner markers
            r = 4
            for px, py in points:
                draw.ellipse((px - r, py - r, px + r, py + r), fill=color)

            # Label
            label_x, label_y = points[0]
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            draw.text((label_x, label_y - 28), obj_data["name"], fill=color, font=font)

        img.save(os.path.join(output_path, fname))
        print(f"Saved {fname}")


if __name__ == "__main__":
    draw_bboxes(JSON_PATH, IMAGES_PATH, OUTPUT_PATH)
