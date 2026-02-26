"""
./src/video_processing/material_tagging/raycasting.py

The full raycasting pipeline, including the camera pose calculator.
"""

import json
import os
from pathlib import Path
from typing import TypedDict

import numpy as np
import open3d as o3d
import pycolmap
from colorama import Fore, init
from PIL import Image

log_path = "logs/video_processing/material_tagging/raycasting.log"

os.makedirs(os.path.dirname(log_path), exist_ok=True)

# sys.stdout = open(log_path, "w")
# sys.stderr = open(log_path, "a")

DATABSE_PATH = "./data/env_imgs/colmap_albert_room/database.db"
IMAGES_PATH = "./data/env_imgs/albert_room"
OUTPUT_PATH = "./data/env_imgs/colmap_albert_room/output"
JSON_PATH = "./data/vision_json/full_detections.json"
OBJ_PATH = "./data/reconstructions/obj/albert_room.obj"
AGGREGATIONS_PATH = "./data/vision_json/aggregations.json"

init(autoreset=True)


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
        debug: bool = False,
    ):
        """
        Initializing the Raycasting class.

        Args:
            database_path (str): Path to COLMAP database file
            image_path (str): Path to input image
            output_path (str): Path to output folder
            json_path (str): Path to the JSON file containing bounding boxes
            obj_path (str): Path to the OBJ mesh file.
            debug (bool): Whether to print debug messages or not. Defaults to False.
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
        self.debug = debug

    def camera_pose(self) -> dict:
        """
        Gets camera poses from database.

        Returns:
            dict: Dictionary with camera poses.
        """
        print(f"{Fore.GREEN}Creating COLMAP map.")

        # maps = pycolmap.incremental_mapping(
        #     database_path=self.db_path,
        #     image_path=self.img_path,
        #     output_path=self.output_path,
        # )

        print(f"{Fore.GREEN}Map has been created.")

        # recon = maps[0]

        poses: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray] | None
        ] = {}  # { image_name: (K, R, t) }

        # for i in range(
        #     len([entry for entry in Path(self.output_path).iterdir() if entry.is_dir()])
        # ):
        #     recon = pycolmap.Reconstruction(os.path.join(self.output_path, str(i)))
        recon_dirs = sorted(
            [entry for entry in Path(self.output_path).iterdir() if entry.is_dir()],
            key=lambda d: d.name,
        )
        best_recon = None
        best_count = 0

        for d in recon_dirs:
            r = pycolmap.Reconstruction(str(d))
            if len(r.images) > best_count:
                best_count = len(r.images)
                best_recon = r

        if best_recon is None:
            raise RuntimeError("No valid COLMAP reconstruction found")

        for img_id, img in best_recon.images.items():
            if self.debug:
                print(best_recon.images)
            cam = best_recon.cameras[img.camera_id]
            if self.debug:
                print(f"{Fore.RED}DEBUG: Image name is: {img.name}")
                # print(f"{Fore.RED}DEBUG: {dir(cam)}")
                # print(f"{Fore.RED}DEBUG: {dir(img.cam_from_world)}")

            fx = cam.focal_length_x
            fy = cam.focal_length_y
            cx = cam.principal_point_x
            cy = cam.principal_point_y
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
            R: np.ndarray = img.cam_from_world().rotation.matrix()
            t: np.ndarray = img.cam_from_world().translation

            if img.name not in poses:  # To avoid duplicates
                poses[img.name] = (K, R, t)

        return poses

    def unprojection(self) -> dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]]:
        """
        Unprojects normalized bounding boxes to the 3D environment as rays

        Returns:
            dict[str, dict[str, list[tuple]]]: Dictionary that represents the rays for each corner of the
                                          bounding box for each object in each frame.
                                          frame_name: {
                                            object_name: [
                                                (origin, ray_world),
                                                (origin, ray_world),
                                                (origin, ray_world),
                                                (origin, ray_world),
                                            ]
                                          }
        """
        print(f"{Fore.GREEN}Starting Unprojection")

        rays = {}
        # object_corner_rays: dict[str, list[tuple]] = {}
        poses = self.camera_pose()
        for frame_name, objects in self.all_frame_detections.items():
            if frame_name not in poses:
                if self.debug:
                    print(f"DEBUG: Skipping {frame_name} (no pose)")
                continue
            rays[frame_name] = {}
            K, R, t = poses[frame_name]
            for obj_name, obj_data in objects.items():
                bbox = obj_data["bounding_box"]
                rays_list = []
                for nx, ny in bbox:
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

                    rays_list.append((origin, ray_world))
                rays[frame_name][obj_name] = rays_list
                if self.debug:
                    print(f"DEBUG: {frame_name}")
        print(f"{Fore.GREEN}Ended Unprojection")
        return rays

    def setup_scene(self) -> o3d.t.geometry.RaycastingScene:
        """
        Loads the OBJ mesh and sets up the Open3D Raycasting Scene

        Returns:
            o3d.t.geometry.RaycastingScene: Open3D raycasting scene with the mesh loaded.
        """
        print(f"{Fore.GREEN}Setting up the scene...")
        mesh_o3d = o3d.io.read_triangle_mesh(self.obj_path)
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)
        return scene

    def raycast(self) -> dict[str, dict[str, list[np.ndarray]]] | None:
        """
        Main raycasting logic.
        hit_points =
        {
            frame_name: {
                object_name:
                    [
                        corner1 rays,
                        corner2 rays,
                        corner3 rays,
                        corner4 rays,
                    ]
            }
        }

        Returns:
            dict[str, dict[str, list[np.ndarray]]]: Dictionary containing all rays for each object.
        """
        print(f"{Fore.GREEN}Starting Raycasting now...")
        rays = self.unprojection()
        scene = self.setup_scene()
        hit_points = {}

        for frame_name, objects in rays.items():
            hit_points[frame_name] = {}
            for (
                obj_name,
                rays_list,
            ) in (
                objects.items()
            ):  # Remember, objects is of this format: {object_name: [(), (), (), ()]}
                hit_points_list = []
                for (
                    origin,
                    direction,
                ) in rays_list:  # Each (origin, direction) tuple is one corner
                    ray = o3d.core.Tensor(
                        [[*origin, *direction]], dtype=o3d.core.float32
                    )
                    result = scene.cast_rays(ray)  # Uses builtin raycasting funtion
                    t_hit = result["t_hit"].numpy()[0]
                    if np.isinf(t_hit):
                        hit_points_list.append(
                            None
                        )  # For every ray that misses, None will be appended
                    else:
                        hit_points_list.append(
                            np.array(origin) + t_hit * np.array(direction)
                        )
                hit_points[frame_name][obj_name] = hit_points_list
        return hit_points

    def aggregate(self) -> dict[str, list]:
        """
        Aggregates 3D hit points for each object across all frames.

        Returns:
            dict[str, list]: The dictionary containing all hit points, top level being each object
        """
        print(f"{Fore.GREEN}Starting aggregation")
        object_points: dict[str, list] = {}
        hit_points = self.raycast()
        if hit_points:
            for _, objects in hit_points.items():
                for hp_obj_name, hit_points_list in objects.items():
                    if hp_obj_name not in object_points:
                        object_points[hp_obj_name] = []
                    object_points[hp_obj_name].extend(
                        [
                            point.tolist()
                            for point in hit_points_list
                            if point is not None
                        ]
                    )  # Convert to list since JSON cannot serialize numpy arrays
        return object_points


def main():
    raycaster = Raycast(
        database_path=DATABSE_PATH,
        images_path=IMAGES_PATH,
        output_path=OUTPUT_PATH,
        json_path=JSON_PATH,
        obj_path=OBJ_PATH,
        debug=False,
    )
    # raycaster.raycast()
    aggregations = raycaster.aggregate()
    # print(aggregations)
    with open(AGGREGATIONS_PATH, "w") as f:
        json.dump(aggregations, f, indent=2)
    print("Aggregations added to JSON. Check data folder.")


if __name__ == "__main__":
    # try:
    #     main()
    # finally:
    #     sys.stdout.close()
    #     sys.stderr.close()
    #     sys.stdout = sys.__stdout__
    #     sys.stderr = sys.__stderr__
    # print(f"{Fore.RED}All debug messages saved to {log_path}")
    main()
