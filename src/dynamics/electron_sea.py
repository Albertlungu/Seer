"""
./src/dynamics/electron_sea.py

Translucent electron-gas cloud for metallic crystal lattices.
"""

import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    Geom,
    GeomNode,
    GeomPoints,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
    TransparencyAttrib,
)


def create_electron_sea(
    base: ShowBase,
    parent: NodePath,
    lattice_positions: np.ndarray,
    lattice_parameter: float,
    color: tuple[float, float, float],
    alpha: float = 0.15,
    points_per_cell: int = 60,
    rng_seed: int = 42,
) -> NodePath:
    """
    Creates a translucent point cloud in the interstitial space of a metal lattice.

    Args:
        base: Panda3D app object.
        parent: Scene-graph parent node.
        lattice_positions: Atom positions of shape (M, 3) in metres.
        lattice_parameter: Unit cell edge length in metres.
        color: RGB tint for the electron cloud.
        alpha: Transparency level (lower = more transparent).
        points_per_cell: Approximate number of cloud points per unit cell.
        rng_seed: Random seed for reproducible sampling.

    Returns:
        NodePath of the electron sea geometry.
    """
    rng = np.random.default_rng(rng_seed)

    bbox_min = lattice_positions.min(axis=0)
    bbox_max = lattice_positions.max(axis=0)
    dims = bbox_max - bbox_min
    volume = float(np.prod(dims))
    cell_volume = lattice_parameter**3
    n_cells_approx = max(1, int(volume / cell_volume))
    total_points = n_cells_approx * points_per_cell

    nn_dist = lattice_parameter * 0.35
    max_candidates = total_points * 5
    candidates = bbox_min + rng.uniform(size=(max_candidates, 3)) * dims

    chunk_size = 5000
    accepted: list[np.ndarray] = []

    for i in range(0, len(candidates), chunk_size):
        chunk = candidates[i : i + chunk_size]
        if len(lattice_positions) > 500:
            sample_idx = rng.choice(len(lattice_positions), 500, replace=False)
            ref_atoms = lattice_positions[sample_idx]
        else:
            ref_atoms = lattice_positions

        diffs = chunk[:, None, :] - ref_atoms[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = dists.min(axis=1)
        mask = min_dists > nn_dist
        accepted.append(chunk[mask])

        if sum(len(a) for a in accepted) >= total_points:
            break

    if not accepted:
        return parent.attachNewNode("empty_sea")

    points = np.concatenate(accepted)[:total_points]

    fmt = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData("electron_sea", fmt, Geom.UHStatic)
    vdata.setNumRows(len(points))
    vertex_writer = GeomVertexWriter(vdata, "vertex")
    color_writer = GeomVertexWriter(vdata, "color")

    for pt in points:
        vertex_writer.addData3f(float(pt[0]), float(pt[1]), float(pt[2]))
        color_writer.addData4f(color[0], color[1], color[2], alpha)

    prim = GeomPoints(Geom.UHStatic)
    for i in range(len(points)):
        prim.addVertex(i)
    prim.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("electron_sea")
    node.addGeom(geom)

    sea_np = parent.attachNewNode(node)
    sea_np.setRenderModeThickness(2.0)
    sea_np.setTransparency(TransparencyAttrib.MAlpha)
    sea_np.setDepthWrite(False)
    return sea_np


def update_electron_sea(
    sea_node: NodePath,
    current_lattice_positions: np.ndarray,
    equilibrium_positions: np.ndarray,
) -> None:
    """
    Shift the electron cloud based on average lattice displacement (phonon motion).

    Args:
        sea_node: The electron sea NodePath.
        current_lattice_positions: Current atom positions, shape (M, 3).
        equilibrium_positions: Original (equilibrium) positions, shape (M, 3).
    """
    if sea_node.isEmpty():
        return
    avg_displacement = np.mean(
        current_lattice_positions - equilibrium_positions, axis=0
    )
    sea_node.setPos(
        float(avg_displacement[0]),
        float(avg_displacement[1]),
        float(avg_displacement[2]),
    )
