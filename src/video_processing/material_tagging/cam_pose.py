"""
./src/video_processing/material_tagging/cam_pose.py

Uses COLMAP to get camera poses from the frames.
"""

from pathlib import Path

import numpy as np
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

    def compute_camera_poses(self):
        maps = pycolmap.incremental_mapping(
            database_path=self.db_path,
            image_path=self.img_path,
            output_path=self.output_path,
        )

        recon = maps[0]

        poses = {}  # { image_name: (K, R, t) }

        for img_id, img in recon.images.items():
            cam = recon.cameras[img.camera_id]

            fx = cam.focal_length_x
            fy = cam.focal_length_y
            cx = cam.principal_point_x
            cy = cam.principal_point_y
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
            R = img.cam_from_world.rotation.matrix()
            t = img.cam_from_world.translation

            poses[img.name] = (K, R, t)


colmap_instance = COLMAP(database_path, image_path, output_path)
