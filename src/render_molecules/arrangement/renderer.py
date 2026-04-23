"""
./src/render_molecules/arrangement/renderer.py

python -m src.render_molecules.arrangement.renderer

Turns arranged molecular states into visible structures. Atoms as normal spheres, but bonds not
as cylinders, but as electron-density regions, reflecting bond order (taking into account sigma/pi bonds)
"""

import math
from typing import cast

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

from src.render_molecules.arrangement.scene_state import (
    MoleculeInstance,
    MoleculeTemplate,
    ObjectState,
)
from src.utils.constants import (
    ANGSTROM_TO_METRES,
    DEFAULT_COLOR,
    DEFAULT_RADIUS,
    ELEMENT_COLORS,
    ELEMENT_RADII,
)
from src.utils.type_annotations import Matrix3x1, Matrix3x3

_DENSITY_NODE_CACHE: dict[tuple, GeomNode] = {}
_ATOM_SCALE_FACTOR: float = 1.0  # Global scale factor for atom rendering
_ALL_ATOM_SPHERES: list[NodePath] = []  # Track all atom spheres for rescaling


def set_atom_scale_factor(scale: float) -> None:
    """
    Set the global atom scale factor and update all existing atoms.

    Args:
        scale: Scale factor (1.0 = van der Waals, ~0.4 = covalent)
    """
    global _ATOM_SCALE_FACTOR
    _ATOM_SCALE_FACTOR = scale

    # Update all existing atom spheres
    for sphere in _ALL_ATOM_SPHERES:
        if sphere and not sphere.isEmpty():
            # Get the base radius from the sphere's tag
            base_radius = sphere.getTag("base_radius")
            if base_radius:
                sphere.setScale(float(base_radius) * scale)


def create_atom_sphere(
    base: ShowBase,
    parent: NodePath,
    template: MoleculeTemplate,
    aid: int,
) -> NodePath:
    """
    Creates a simple sphere representing one atom.

    Args:
        base (ShowBase): Panda3D app object.
        parent (NodePath): Scene-graph parent.
        template (MoleculeTemplate): Source template.
        aid (int): Atom ID to locate in the template.

    Raises:
        ValueError: If aid does not match one entry in the template
        RuntimeError: If base.loader is not initialized

    Returns:
        NodePath: The positioned sphere Node.
    """
    atom_idx_array = np.where(template.aids == aid)[0]
    if len(atom_idx_array) != 1:
        raise ValueError(f"Expected 1 match, got {len(atom_idx_array)}")
    atom_idx = int(atom_idx_array[0])

    element = template.elements[atom_idx]
    color = ELEMENT_COLORS.get(element, DEFAULT_COLOR)

    # Use van der Waals radii as base, convert to Angstroms
    base_radius = ELEMENT_RADII.get(element, DEFAULT_RADIUS) / ANGSTROM_TO_METRES

    if base.loader is None:
        raise RuntimeError("ShowBase loader not initialized")
    sphere = cast(NodePath, base.loader.loadModel("models/misc/sphere"))
    sphere.setName(f"atom_{aid}")

    sphere.reparentTo(parent)
    x, y, z = np.column_stack(template.local_xyz)[atom_idx]
    sphere.setPos(float(x), float(y), float(z))

    # Store base radius in tag for dynamic rescaling
    sphere.setTag("base_radius", str(base_radius))

    # Apply current scale factor
    sphere.setScale(base_radius * _ATOM_SCALE_FACTOR)
    sphere.setColor(*color, 1.0)

    # Track this sphere for dynamic rescaling
    _ALL_ATOM_SPHERES.append(sphere)

    return sphere


def build_instance_root(
    parent: NodePath,
    template: MoleculeTemplate,
    instance: MoleculeInstance,
    base: ShowBase,
) -> NodePath:
    """
    Attaches atom spheres and bond clouds for one molecule instance, then sets its world pose.

    Args:
        parent (NodePath): Scene-graph parent
        template (MoleculeTemplate): Source template
        instance (MoleculeInstance): Instance carrying position and hpr
        base (ShowBase): Panda3D app object

    Returns:
        NodePath: The molecule root node with pose applied
    """
    root = parent.attachNewNode(f"molecule{instance.id}")
    local_coords = np.column_stack(template.local_xyz)
    aid_to_index = {int(aid): idx for idx, aid in enumerate(template.aids)}

    for aid in template.aids:
        create_atom_sphere(base=base, parent=root, template=template, aid=int(aid))

    for aid1, aid2, order in zip(
        template.bonds_aid1, template.bonds_aid2, template.bond_order
    ):
        idx1 = aid_to_index.get(int(aid1))
        idx2 = aid_to_index.get(int(aid2))
        if idx1 is None or idx2 is None:
            continue
        create_bond_visual(
            base=base,
            parent=root,
            atom_a=local_coords[idx1],
            atom_b=local_coords[idx2],
            bond_order=int(order),
        )

    h, p, r = instance.hpr
    root.setPos(*instance.position.tolist())
    root.setHpr(math.degrees(h), math.degrees(p), math.degrees(r))
    return root


def render_object_state(
    base: ShowBase, parent: NodePath, object_state: ObjectState
) -> dict[int, NodePath]:
    """
    Builds render nodes for all placed instances in one object state.

    Args:
        base (ShowBase): Panda3D app object
        parent (NodePath): Scene-graph parent for all molecule roots
        object_state (ObjectState): Fully placed object state

    Returns:
        dict[int, NodePath]: Maps instance ID to molecule root node
    """
    instance_roots: dict[int, NodePath] = {}
    for instance_id, instance in object_state.instances.items():
        template = object_state.templates[instance.template_id]
        instance_roots[instance_id] = build_instance_root(
            parent=parent, template=template, instance=instance, base=base
        )
    return instance_roots


def clear_removed_instance(
    instance_roots: dict[int, NodePath],
    instance_id: int,
) -> None:
    """
    Removes one rendered molecule instance by ID

    Args:
        instance_roots (dict[int, NodePath]): Cache of rendered instance roots
        instance_id (int): Instance ID to remove
    """
    root = instance_roots.pop(instance_id, None)
    if root is not None:
        root.removeNode()


def sync_scene_render(
    base: ShowBase,
    parent: NodePath,
    object_state: ObjectState,
    instance_roots: dict[int, NodePath],
) -> dict[int, NodePath]:
    """
    Synchronizes render nodes with the current object state.

    Args:
        base (ShowBase): Panda3D app object
        parent (NodePath): Scene-graph parent
        object_state (ObjectState): Current arranged state
        instance_roots (dict[int, NodePath]): Existing cache of rendered roots

    Returns:
        dict[int, NodePath]: Updated instance root map
    """
    live_ids = set(object_state.instances.keys())
    cached_ids = set(instance_roots.keys())

    for removed_id in cached_ids - live_ids:
        clear_removed_instance(instance_roots=instance_roots, instance_id=removed_id)

    for instance_id in live_ids:
        instance = object_state.instances[instance_id]
        if instance_id not in instance_roots:
            template = object_state.templates[instance.template_id]
            instance_roots[instance_id] = build_instance_root(
                parent=parent, template=template, instance=instance, base=base
            )
            continue

        root = instance_roots[instance_id]
        h, p, r = instance.hpr
        root.setPos(*instance.position.tolist())
        root.setHpr(math.degrees(h), math.degrees(p), math.degrees(r))

        return instance_roots


def bond_order_to_vis(bond_order: int) -> tuple[int, int]:
    """
    Maps bond order to (sigma_count, pi_count). Does not account for delta bonds.

    Args:
        bond_order (int): The bond order

    Returns:
        tuple[int, int]: Number of sigma components, number of pi components
    """
    if bond_order <= 1:
        return 1, 0
    if bond_order == 2:
        return 1, 1
    return 1, 2


def _perpendicular_axes(unit_axis: Matrix3x1) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds two vectors perpendicular to a bond axis using Gram-Schmidt.

    Args:
        unit_axis (Matrix3x1): Unit vector along the bond axis.

    Returns:
        tuple[np.ndarray, np.ndarray]: Two orthonormal vectors perpendicular to unit_axis.
    """
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(unit_axis, ref))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])

    u = np.cross(unit_axis, ref)
    u_norm = float(np.linalg.norm(u))
    if u_norm < 1e-8:
        u = np.array([1.0, 0.0, 0.0])
        u_norm = 1.0
    u /= u_norm

    v = np.cross(unit_axis, u)
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-8:
        v = np.array([0.0, 1.0, 0.0])
        v_norm = 1.0
    v /= v_norm

    return u, v


def _create_density_cloud(
    base: ShowBase,
    parent: NodePath,
    start: Matrix3x1,
    end: Matrix3x1,
    center_offset: Matrix3x1,
    radial_scale: float,
    alpha: float,
    color: tuple[float, float, float],
    lobe_axis: np.ndarray | None = None,
) -> NodePath:
    """
    Creates a sampled dot cloud from orbital-density formulas.

    Args:
        base (ShowBase): Panda3D app object.
        parent (NodePath): Node parent.
        start (Matrix3x1): Starting atom position.
        end (Matrix3x1): Ending atom position.
        center_offset (Matrix3x1): Offset from bond midpoint (zero for sigma, +/- for pi lobes).
        radial_scale (float): Cloud thickness perpendicular to bond axis.
        alpha (float): Transparency level.
        color (tuple[float, float, float]): RGB color for dots.
        lobe_axis (np.ndarray | None): If given, uses a pi-type orbital term along this axis.

    Raises:
        RuntimeError: If base.loader is not initialized.

    Returns:
        NodePath: The electron cloud node.
    """
    start_vec = np.asarray(start, dtype=float).reshape(3)
    end_vec = np.asarray(end, dtype=float).reshape(3)
    offset_vec = np.asarray(center_offset, dtype=float).reshape(3)

    axis = end_vec - start_vec
    length = float(np.linalg.norm(axis))
    if length < 1e-8:
        length = 1e-8
    axis = axis / length

    mid = ((start_vec + end_vec) * 0.5) + offset_vec
    u, v = _perpendicular_axes(axis)

    seed_source = np.concatenate([start_vec, end_vec, offset_vec])
    seed = int(np.sum(np.abs(seed_source) * 1_000_000.0)) % (2**32 - 1)
    if seed == 0:
        seed = 1

    # Increased dot count for denser electron clouds
    dot_count = max(500, min(2000, int(800 + (3000 * radial_scale) + (200 * length))))
    dot_radius = max(0.003, radial_scale * 0.17)

    lobe_key = (
        None
        if lobe_axis is None
        else tuple(np.round(np.asarray(lobe_axis, dtype=float).reshape(3), 5))
    )
    cache_key = (
        tuple(np.round(start_vec, 5)),
        tuple(np.round(end_vec, 5)),
        tuple(np.round(offset_vec, 5)),
        round(radial_scale, 5),
        round(alpha, 5),
        tuple(round(c, 5) for c in color),
        lobe_key,
        dot_count,
    )

    if cache_key in _DENSITY_NODE_CACHE:
        cloud = NodePath(_DENSITY_NODE_CACHE[cache_key]).copyTo(parent)
        cloud.setRenderModeThickness(max(1.0, dot_radius * 120.0))
        cloud.setTransparency(TransparencyAttrib.MAlpha)
        cloud.setDepthWrite(False)
        return cloud

    rng = np.random.default_rng(seed)

    zeta = max(2.0 / max(length, 0.2), 1.25)
    half_len = 0.5 * length
    radial_extent = max(radial_scale * 4.5, 0.14)
    along_extent = half_len + max(0.22 * length, 0.08)
    candidate_count = dot_count * 9

    if lobe_axis is None:
        orbital_axis = None
    else:
        orbital_axis = np.asarray(lobe_axis, dtype=float).reshape(3)
        orbital_norm = float(np.linalg.norm(orbital_axis))
        if orbital_norm < 1e-8:
            orbital_axis = None
        else:
            orbital_axis = orbital_axis / orbital_norm

    along = rng.uniform(-along_extent, along_extent, size=candidate_count)
    off_u = rng.uniform(-radial_extent, radial_extent, size=candidate_count)
    off_v = rng.uniform(-radial_extent, radial_extent, size=candidate_count)
    candidates = (
        mid[None, :]
        + along[:, None] * axis[None, :]
        + off_u[:, None] * u[None, :]
        + off_v[:, None] * v[None, :]
    )

    r_a_vec = candidates - start_vec[None, :]
    r_b_vec = candidates - end_vec[None, :]
    r_a = np.linalg.norm(r_a_vec, axis=1)
    r_b = np.linalg.norm(r_b_vec, axis=1)
    phi_a = np.exp(-zeta * r_a)
    phi_b = np.exp(-zeta * r_b)

    if orbital_axis is None:
        psi = phi_a + phi_b
    else:
        psi = (r_a_vec @ orbital_axis) * phi_a + (r_b_vec @ orbital_axis) * phi_b

    rho = psi * psi
    rho_sum = float(np.sum(rho))
    if rho_sum <= 1e-20:
        weights = np.full(candidate_count, 1.0 / candidate_count)
    else:
        weights = rho / rho_sum

    chosen = rng.choice(candidate_count, size=dot_count, replace=True, p=weights)
    points = candidates[chosen]

    fmt = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData("density_points", fmt, Geom.UHStatic)
    vdata.setNumRows(dot_count)
    vertex_writer = GeomVertexWriter(vdata, "vertex")
    color_writer = GeomVertexWriter(vdata, "color")

    for point in points:
        vertex_writer.addData3f(float(point[0]), float(point[1]), float(point[2]))
        color_writer.addData4f(color[0], color[1], color[2], alpha)

    prim = GeomPoints(Geom.UHStatic)
    for i in range(dot_count):
        prim.addVertex(i)
    prim.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("density_points")
    node.addGeom(geom)
    _DENSITY_NODE_CACHE[cache_key] = node

    cloud = NodePath(node).copyTo(parent)
    cloud.setRenderModeThickness(max(1.0, dot_radius * 120.0))
    cloud.setTransparency(TransparencyAttrib.MAlpha)
    cloud.setDepthWrite(False)
    return cloud


def create_sigma_bond_cloud(
    base: ShowBase,
    parent: NodePath,
    atom_a: np.ndarray,
    atom_b: np.ndarray,
    bond_order: int,
) -> NodePath:
    """
    Creates one sigma bond cloud aligned with the bond axis.

    Args:
        base (ShowBase): Panda3D app object.
        parent (NodePath): Node parent.
        atom_a (np.ndarray): First atom position in local molecule space.
        atom_b (np.ndarray): Second atom position in local molecule space.
        bond_order (int): Bond order used to scale sigma cloud thickness.

    Returns:
        NodePath: The sigma electron-density cloud node.
    """
    sigma_radius = 0.05 + (0.01 * max(0, bond_order - 1))
    return _create_density_cloud(
        base=base,
        parent=parent,
        start=atom_a,
        end=atom_b,
        center_offset=np.zeros(3, dtype=float),
        radial_scale=sigma_radius,
        alpha=0.45,
        color=(0.72, 0.72, 0.72),
        lobe_axis=None,
    )


def create_pi_bond_cloud(
    base: ShowBase,
    parent: NodePath,
    atom_a: np.ndarray,
    atom_b: np.ndarray,
    offset_axis: np.ndarray,
) -> tuple[NodePath, NodePath]:
    """
    Creates the two opposite lobes for one pi bond component.

    Args:
        base (ShowBase): Panda3D app object.
        parent (NodePath): Node parent.
        atom_a (np.ndarray): First atom position in local molecule space.
        atom_b (np.ndarray): Second atom position in local molecule space.
        offset_axis (np.ndarray): Unit axis perpendicular to the bond axis for lobe offsets.

    Returns:
        tuple[NodePath, NodePath]: Positive and negative pi lobe nodes.
    """
    lobe_offset = np.asarray(offset_axis, dtype=float).reshape(3) * 0.12
    pos_lobe = _create_density_cloud(
        base=base,
        parent=parent,
        start=atom_a,
        end=atom_b,
        center_offset=lobe_offset,
        radial_scale=0.035,
        alpha=0.38,
        color=(0.42, 0.66, 1.0),
        lobe_axis=offset_axis,
    )
    neg_lobe = _create_density_cloud(
        base=base,
        parent=parent,
        start=atom_a,
        end=atom_b,
        center_offset=-lobe_offset,
        radial_scale=0.035,
        alpha=0.38,
        color=(0.42, 0.66, 1.0),
        lobe_axis=offset_axis,
    )
    return pos_lobe, neg_lobe


def create_bond_visual(
    base: ShowBase,
    parent: NodePath,
    atom_a: np.ndarray,
    atom_b: np.ndarray,
    bond_order: int,
) -> list[NodePath]:
    """
    Creates all bond cloud nodes for one covalent bond.

    Args:
        base (ShowBase): Panda3D app object.
        parent (NodePath): Node parent.
        atom_a (np.ndarray): First atom position in local molecule space.
        atom_b (np.ndarray): Second atom position in local molecule space.
        bond_order (int): Bond order.

    Returns:
        list[NodePath]: All created cloud nodes for the bond.
    """
    visuals: list[NodePath] = []
    sigma_count, pi_count = bond_order_to_vis(int(bond_order))

    if sigma_count:
        visuals.append(
            create_sigma_bond_cloud(
                base=base,
                parent=parent,
                atom_a=atom_a,
                atom_b=atom_b,
                bond_order=int(bond_order),
            )
        )

    axis = np.asarray(atom_b, dtype=float).reshape(3) - np.asarray(
        atom_a, dtype=float
    ).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        return visuals
    axis = axis / norm
    u, v = _perpendicular_axes(axis)

    if pi_count >= 1:
        visuals.extend(create_pi_bond_cloud(base, parent, atom_a, atom_b, u))
    if pi_count >= 2:
        visuals.extend(create_pi_bond_cloud(base, parent, atom_a, atom_b, v))


def update_atom_positions(
    instance_roots: dict[int, NodePath],
    atom_mapping: "AtomMapping",
    positions: np.ndarray,
    object_state: ObjectState,
) -> None:
    """
    Move existing atom sphere NodePaths to new positions from the simulation buffer.
    Does not recreate the scene graph.

    Args:
        instance_roots: Map of instance ID to molecule root NodePath.
        atom_mapping: AtomMapping from the simulation thread.
        positions: Flat (N, 3) array of current atom positions in world space.
        object_state: Current object state for template lookup.
    """
    for instance_id, (start, end) in atom_mapping.instance_to_sim_range.items():
        root = instance_roots.get(instance_id)
        if root is None or root.isEmpty():
            continue

        atom_nodes = sorted(
            [ch for ch in root.getChildren() if ch.getName().startswith("atom_")],
            key=lambda n: int(n.getName().split("_")[1]),
        )

        n_expected = end - start
        if len(atom_nodes) != n_expected:
            continue

        # Buffer is in metres; NodePaths live in mol_root local space (Angstroms).
        inst_positions_a = positions[start:end] / 1e-10  # metres -> Angstroms

        root_pos = root.getPos(root.getParent())
        rx, ry, rz = float(root_pos.x), float(root_pos.y), float(root_pos.z)

        for node, pos_a in zip(atom_nodes, inst_positions_a):
            node.setPos(
                float(pos_a[0]) - rx,
                float(pos_a[1]) - ry,
                float(pos_a[2]) - rz,
            )


def rebuild_bond_clouds(
    instance_roots: dict[int, NodePath],
    object_state: ObjectState,
    base: ShowBase,
) -> None:
    """
    Remove existing bond clouds and recreate them at current atom positions.
    Only call when cloud rendering is enabled during dynamics.

    Args:
        instance_roots: Map of instance ID to molecule root NodePath.
        object_state: Current object state for template/bond lookup.
        base: Panda3D app instance.
    """
    for instance_id, root in instance_roots.items():
        if root is None or root.isEmpty():
            continue

        inst = object_state.instances.get(instance_id)
        if inst is None:
            continue
        template = object_state.templates[inst.template_id]

        for child in root.getChildren():
            if not child.getName().startswith("atom_"):
                child.removeNode()

        atom_nodes = sorted(
            [ch for ch in root.getChildren() if ch.getName().startswith("atom_")],
            key=lambda n: int(n.getName().split("_")[1]),
        )

        if len(atom_nodes) != len(template.aids):
            continue

        local_coords = np.array(
            [[float(n.getX()), float(n.getY()), float(n.getZ())] for n in atom_nodes]
        )

        aid_to_index = {int(aid): idx for idx, aid in enumerate(template.aids)}

        for aid1, aid2, order in zip(
            template.bonds_aid1, template.bonds_aid2, template.bond_order
        ):
            idx1 = aid_to_index.get(int(aid1))
            idx2 = aid_to_index.get(int(aid2))
            if idx1 is None or idx2 is None:
                continue
            create_bond_visual(
                base=base,
                parent=root,
                atom_a=local_coords[idx1],
                atom_b=local_coords[idx2],
                bond_order=int(order),
            )

    return visuals
