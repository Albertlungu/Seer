"""
./src/video_processing/material_tagging/cam_pose.py

Given an input folder of images, generates a COLMAP database.
"""

from pathlib import Path

import pycolmap

database_path = Path("./data/env_imgs/colmap_albert_room/database.db")
image_path = Path("./data/env_imgs/albert_room")
output_path = Path("./data/env_imgs/colmap_albert_room/output")


class COLMAP:
    def __init__(self, db_path, img_path, output_path):
        self.db_path = db_path
        self.img_path = img_path
        self.output_path = output_path

    def generate_database(self):
        pycolmap.extract_features(self.db_path, self.img_path)
        pairing = pycolmap.SequentialPairingOptions(overlap=10)
        pycolmap.match_sequential(self.db_path, pairing_options=pairing)

    # generate_database(database_path, image_path)


colmap_instance = COLMAP(database_path, image_path, output_path)
