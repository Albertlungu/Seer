import os
import subprocess


def extract_frames(vid_path, out_folder, fps=2):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    command = [
        "ffmpeg",
        "-i",
        vid_path,
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        f"{out_folder}/frame_%04d.jpg",
    ]

    print(f"Extracting frames to {out_folder}...")
    subprocess.run(command)
    print("Extraction complete.")


extract_frames("data/env_vids/albert_room.mov", "data/env_imgs/albert_room")
