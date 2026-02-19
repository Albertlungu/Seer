import argparse
import os
import zipfile

import aspose.threed as a3d
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-file", default="data/reconstructions/usdz/albert_room.usdz"
)
parser.add_argument("--output-file", default="data/reconstructions/obj/albert_room.obj")
args = parser.parse_args()


def extract_textures_from_usdz(usdz_path, output_dir):
    """USDZ is a zip archive -- extract any texture images from it."""
    with zipfile.ZipFile(usdz_path, "r") as z:
        for name in z.namelist():
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                # Extract to output directory with just the filename (no subdirs)
                basename = os.path.basename(name)
                target = os.path.join(output_dir, basename)
                with z.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                print(f"  Extracted texture: {basename}")


def convert_to_obj(input_file, output_file):
    if ".obj" in input_file:
        print("Was already an OBJ")
    elif ".usdz" in input_file:
        print(f"Converting to .obj to {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        scene = a3d.Scene.from_file(input_file)
        scene.save(output_file, a3d.FileFormat.WAVEFRONT_OBJ)
        # Extract textures that the MTL references
        extract_textures_from_usdz(input_file, os.path.dirname(output_file))
        print("Done")
    else:
        print("Was not one of either OBJ or USDZ. Oops...")


convert_to_obj(args.input_file, args.output_file)


mesh = o3d.io.read_triangle_mesh(args.output_file, enable_post_processing=True)
mesh.compute_vertex_normals()

if not mesh.has_triangle_uvs():
    print("Warning: No UVs found in mesh")
if not mesh.textures:
    print("Warning: No textures loaded")

o3d.visualization.draw_geometries(
    [mesh], mesh_show_wireframe=False, mesh_show_back_face=True
)
