"""
./src/video_processing/material_tagging/raycasting.py

The full raycasting pipeline, including the camera pose calculator.
"""

from pathlib import Path

import numpy as np
import open3d as o3d
import pycolmap

database_path = Path("./data/env_imgs/colmap_albert_room/database.db")
image_path = Path("./data/env_imgs/albert_room")
output_path = Path("./data/env_imgs/colmap_albert_room/output")


class Raycast:
    def __init__(
        self,
        database_path: str,
        image_path: str,
        output_path: str,
        bbox: list[tuple],
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        image_w: int,
        image_h: int,
        obj_path: str,
        origin: np.ndarray,
        direction: np.ndarray,
    ):
        """
        Initializing the Raycasting class.

        Args:
            database_path (str): Path to COLMAP database file
            image_path (str): Path to input image
            output_path (str): Path to output folder
            bbox (list[tuple]): List of 4 (nx, ny) tuples. The normalized [0, 1] coordinates.
            K (np.ndarray): Intrinsic matrix (3x3 array)
            R (np.ndarray): Rotation matrix (3x3 array)
            t (np.ndarray): Translation vector (3x1 array)
            image_w (int): Width of the image in pixels
            image_h (int): Height of the image in pixels
            obj_path (str): Path to the OBJ mesh file.
            origin: (np.ndarray): 3D coordinates of the ray origin
            direction (np.ndarray): 3D vector for the ray direction
        """
        self.bbox = bbox
        self.K = K
        self.R = R
        self.t = t
        self.image_w = image_w
        self.image_h = image_h
        self.db_path = database_path
        self.img_path = image_path
        self.output_path = output_path
        self.obj_path = obj_path
        self.origin = origin

    def camera_pose(self) -> dict:
        """
        Gets camera poses from database.

        Returns:
            dict: Dictionary with camera poses.
        """
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

        return poses

    def unprojection(self) -> list:
        """
        Unprojects normalized bounding boxes to the 3D environment as rays

        Returns:
            list: List of (origin, direction) tuples for each corner of the bounding box
        """
        rays = []
        for nx, ny in self.bbox:
            u = (
                nx * self.image_w
            )  # The actual pixel in the image corresponding to the Bbox corner
            v = ny * self.image_h
            pixel_h = np.array([u, v, 1.0])

            ray_cam = np.linalg.inv(K) @ pixel_h
            ray_cam /= np.linalg.norm(ray_cam)

            ray_world = self.R.T @ ray_cam
            ray_world /= np.linalg.norm(ray_world)

            origin = -self.R.T @ self.t

            rays.append((origin, ray_world))

        return rays

    def setup_scene(self) -> o3d.t.geometry.RaycastingScene:
        """
        Loads the OBJ mesh and sets up the Open3D Raycasting Scene

        Returns:
            o3d.t.geometry.RaycastingScene: Open3D raycasting scene with the mesh loaded.
        """
        mesh_o3d = o3d.io.read_triangle_mesh(self.obj_path)
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)
        return scene

    def raycast(self) -> list | None:
        """
        Main raycasting logic.

        Returns:
            list | None: list of 3D hit points on mesh or None if all rays.
        """
        rays = self.unprojection()
        scene = self.setup_scene()
        hit_points = []

        for origin, direction in rays:
            ray = o3d.core.Tensor([[*origin, *direction]], dtype=o3d.core.float32)
            result = scene.cast_rays(ray)
            t_hit = result["t_hit"].numpy()[0]
            if np.isinf(t_hit):
                hit_points.append(None)
            else:
                hit_points.append(np.array(origin) + t_hit * np.array(direction))
        return hit_points
