"""
./src/video_processing/material_tagging/raycasting.py

The full raycasting pipeline, including the camera pose calculator.
"""

import json
from typing import TypedDict

import numpy as np
import open3d as o3d
import pycolmap
from PIL import Image

DATABSE_PATH = "./data/env_imgs/colmap_albert_room/database.db"
IMAGES_PATH = "./data/env_imgs/albert_room"
OUTPUT_PATH = "./data/env_imgs/colmap_albert_room/output"
JSON_PATH = "./data/vision_json/albert_room.json"
OBJ_PATH = "./data/reconstructions/obj/albert_room.obj"


class ObjectData(TypedDict):
    name: str
    materials: list[str]
    bounding_box: list[list[float]]


Detections = dict[str, dict[str, ObjectData]]


class Raycast:
    def __init__(
        self,
        database_path: str,
        images_path: str,
        json_path: str,
        output_path: str,
        # bbox: list[tuple],
        # K: np.ndarray,
        # R: np.ndarray,
        # t: np.ndarray,
        obj_path: str,
    ):
        """
        Initializing the Raycasting class.

        Args:
            database_path (str): Path to COLMAP database file
            image_path (str): Path to input image
            output_path (str): Path to output folder
            json_path (str): Path to the JSON file containing bounding boxes
            obj_path (str): Path to the OBJ mesh file.
        """
        self.img_path = images_path
        self.output_path = output_path
        self.json_path = json_path
        self.obj_path = obj_path
        with open(self.json_path, "rb") as f:
            self.all_frame_detections: Detections = json.load(f)
        # self.K = K
        # self.R = R
        # self.t = t
        img = Image.open(self.img_path + "/frame_0001.jpg")
        self.image_w, self.image_h = img.size
        self.db_path = database_path

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

        poses: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray] | None
        ] = {}  # { image_name: (K, R, t) }

        for (
            img_id,
            img,
        ) in (
            recon.images.items()
        ):  # recon.images: dict, img_id: str (the image name), img: Image
            cam = recon.cameras[img.camera_id]

            fx = cam.focal_length_x
            fy = cam.focal_length_y
            cx = cam.principal_point_x
            cy = cam.principal_point_y
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
            R: np.ndarray = img.cam_from_world.rotation.matrix()
            t: np.ndarray = img.cam_from_world.translation

            poses[img.name] = (K, R, t)

        return poses

    def unprojection(self) -> list:
        """
        Unprojects normalized bounding boxes to the 3D environment as rays

        Returns:
            list: List of (origin, direction) tuples for each corner of the bounding box
        """
        rays = []
        bbox_temp = []
        for object in self.all_frame_detections.values():
            poses = self.camera_pose()
            for pose in poses.values():
                K: np.ndarray = pose[0]
                R: np.ndarray = pose[1]
                t: np.ndarray = pose[2]
            bbox = object["bounding_box"]
            for coord in bbox:
                for nx, ny in coord:
                    u = (
                        nx * self.image_w
                    )  # The actual pixel in the image corresponding to the Bbox corner
                    v = ny * self.image_h
                    pixel_h = np.array([u, v, 1.0])

                    ray_cam = np.linalg.inv(K) @ pixel_h
                    ray_cam /= np.linalg.norm(ray_cam)

                    ray_world = R.T @ ray_cam
                    ray_world /= np.linalg.norm(ray_world)

                    origin = -R.T @ t

                    temp = (origin, ray_world)
                bbox_temp.append(temp)
            rays.append(bbox_temp)
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
            result = scene.cast_rays(ray)  # Uses builtin raycasting funtion
            t_hit = result["t_hit"].numpy()[0]
            if np.isinf(t_hit):
                hit_points.append(
                    None
                )  # For every ray that misses, None will be appended
            else:
                hit_points.append(np.array(origin) + t_hit * np.array(direction))
        return hit_points

    def aggregation(self):
        poses = self.camera_pose()


raycaster = Raycast(
    database_path=DATABSE_PATH,
    images_path=IMAGES_PATH,
    output_path=OUTPUT_PATH,
    json_path=JSON_PATH,
    obj_path=OBJ_PATH,
)
print(raycaster.all_frame_detections)
