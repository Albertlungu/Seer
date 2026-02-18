"""
./src/video_processing/reconstruction/vid_to_imgs.py

Turns a video into a folder of images.
"""

import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument(
    "--vid-path",
    default="data/env_vids/albert_room.mov",
    help="The relative path to your environment video",
)
parser.add_argument(
    "--output-folder",
    default="data/env_imgs/albert_room",
    help="Output file for the imgs folder",
)

args = parser.parse_args()


def extract_frames(vid_path: str, output_folder: str, fps=2) -> None:
    """
    Uses MacOS's FFMPEG library to turn a video into a folder of images.

    Args:
        vid_path (str): Path to the video.
        out_folder (str): Path to the output folder of images.
        fps (int, optional): At how many frames the system takes an image. Defaults to 2.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    command = [
        "ffmpeg",
        "-i",
        vid_path,
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        f"{output_folder}/frame_%04d.jpg",
    ]

    print(f"Extracting frames to {output_folder}...")
    subprocess.run(command)
    print("Extraction complete.")


extract_frames(args.vid_path, args.output_folder)
