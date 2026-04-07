"""
./src/render_molecules/arrangement/renderer.py

python -m src.render_molecules.arrangement.renderer

Turns arranged molecular states into visible structures. Atoms as normal spheres, but bonds not
as cylinders, but as electron-density regions, reflecting bond order (taking into account sigma/pi bonds)
"""

from typing import cast

import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Mat4, NodePath, TransparencyAttrib

from src.render_molecules.arrangement.scene_state import (
    MoleculeInstance,
    MoleculeTemplate,
    ObjectState,
)
from src.utils.constants import (
    DEFAULT_COLOR,
    DEFAULT_RADIUS,
    ELEMENT_COLORS,
    ELEMENT_RADII,
)
from src.utils.type_annotations import Matrix3x1, Matrix3x3


def create_atom_sphere(
    base: ShowBase,
    parent: NodePath,
    template: MoleculeTemplate,
    aid: int,
) -> NodePath:
    """
    Creates a sphere representing an atom.

    Args:
        base (ShowBase): Panda3D app object
        parent (NodePath): Scene-graph parent
        template (MoleculeTemplate): Target molecule template
        aid (int): Atom ID

    Raises:
        ValueError: If there are no matching aids
        RuntimeError: If the base loader is not initialized

    Returns:
        NodePath: The sphere.
    """

    atom_idx_array = np.where(template.aids == aid)[0]
    if len(atom_idx_array) != 1:
        raise ValueError(f"Expected 1 matching aid, got {len(atom_idx_array)}")
    atom_idx = int(atom_idx_array[0])

    element = template.elements[atom_idx]
    color = ELEMENT_COLORS.get(element, DEFAULT_COLOR)
    radius = ELEMENT_RADII.get(element, DEFAULT_RADIUS)

    if base.loader is None:
        raise RuntimeError("ShowBase loader is not initialized")
    sphere = cast(NodePath, base.loader.loadModel("models/misc/sphere"))

    sphere.reparentTo(parent)
    x, y, z = np.column_stack(template.local_xyz)[atom_idx]
    sphere.setPos(float(x), float(y), float(z))
    sphere.setScale(radius)
    sphere.setColor(*color, 1.0)

    return sphere


def _pose_to_mat4(rotation: Matrix3x3, position: Matrix3x1) -> Mat4:
    """
    Converts position and rotation (pose) to a Panda3D mat4 object

    Args:
        rotation (Matrix3x3): Rotation matrix, shape (3, 3)
        position (Matrix3x1): Position matrix, shape (3,)

    Returns:
        Mat4: The converted Mat4 object representing both rotation and position
    """
    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    t = np.asarray(position, dtype=float).reshape(3)

    return Mat4(
        r[0, 0],
        r[0, 1],
        r[0, 2],
        0.0,
        r[1, 0],
        r[1, 1],
        r[1, 2],
        0.0,
        r[2, 0],
        r[2, 1],
        r[2, 2],
        0.0,
        t[0],
        t[1],
        t[2],
        1.0,
    )


def build_instance_root(
    parent: NodePath,
    template: MoleculeTemplate,
    instance: MoleculeInstance,
    base: ShowBase,
) -> NodePath:
    """
    Builds the instance root to handle all the translation and rotation for the molecule as a whole.

    Args:
        parent (NodePath): Node parent.
        template (MoleculeTemplate): Molecule template
        instance (MoleculeInstance): Molecule instance
        base (ShowBase): Panda3D base app object

    Returns:
        NodePath: The molecule
    """
    root = parent.attachNewNode(f"molecule_{instance.id}")
    local_coords = np.column_stack(template.local_xyz)
    aid_to_index = {int(aid): idx for idx, aid in enumerate(template.aids)}

    for aid in template.aids:
        create_atom_sphere(base=base, parent=root, template=template, aid=int(aid))

    for aid1, aid2, order in zip(
        template.bonds_aid1,
        template.bonds_aid2,
        template.bond_order,
    ):
        idx1 = aid_to_index.get(int(aid1))
        idx2 = aid_to_index.get(int(aid2))
        if idx1 is None or idx2 is None:
            continue

        atom_a = local_coords[idx1]
        atom_b = local_coords[idx2]
        create_bond_visual(
            base=base,
            parent=root,
            atom_a=atom_a,
            atom_b=atom_b,
            bond_order=int(order),
        )

    root.setMat(_pose_to_mat4(instance.rotation, instance.position))
    return root


def render_object_state(
    base: ShowBase,
    parent: NodePath,
    object_state: ObjectState,
) -> dict[int, NodePath]:
    """Build render nodes for all placed instances in an object state."""
    instance_roots: dict[int, NodePath] = {}
    for instance_id, instance in object_state.instances.items():
        template = object_state.templates[instance.template_id]
        instance_roots[instance_id] = build_instance_root(
            parent=parent,
            template=template,
            instance=instance,
            base=base,
        )
    return instance_roots


def clear_removed_instance(
    instance_roots: dict[int, NodePath],
    instance_id: int,
) -> None:
    """Remove one rendered molecule instance by id."""
    root = instance_roots.pop(instance_id, None)
    if root is not None:
        root.removeNode()


def sync_scene_render(
    base: ShowBase,
    parent: NodePath,
    object_state: ObjectState,
    instance_roots: dict[int, NodePath],
) -> dict[int, NodePath]:
    """Update/add/remove molecule roots to match the current object state."""
    live_ids = set(object_state.instances.keys())
    cached_ids = set(instance_roots.keys())

    for removed_id in cached_ids - live_ids:
        clear_removed_instance(instance_roots=instance_roots, instance_id=removed_id)

    for instance_id in live_ids:
        instance = object_state.instances[instance_id]
        if instance_id not in instance_roots:
            template = object_state.templates[instance.template_id]
            instance_roots[instance_id] = build_instance_root(
                parent=parent,
                template=template,
                instance=instance,
                base=base,
            )
            continue

        instance_roots[instance_id].setMat(
            _pose_to_mat4(instance.rotation, instance.position)
        )

    return instance_roots


def bond_order_to_vis(bond_order: int) -> tuple[int, int]:
    """
    Maps bond order to visual component counts. Does not account for delta bonds.

    Args:
        bond_order (int): The bond order.

    Returns:
        tuple[int, int]: number of sigma bonds, number of pi bonds
    """
    if bond_order <= 1:
        return 1, 0
    if bond_order == 2:
        return 1, 1
    return 1, 2


def _create_density_cloud(
    base: ShowBase,
    parent: NodePath,
    start: Matrix3x1,
    end: Matrix3x1,
    center_offset: Matrix3x1,
    radial_scale: float,
    alpha: float,
    color: tuple[float, float, float],
) -> NodePath:
    """
    Creates a density cloud for electron bonds.

    Args:
        base (ShowBase): Panda3D Base app object
        parent (NodePath): Node parent
        start (Matrix3x1): Starting point (one atom)
        end (Matrix3x1): Ending point (the other atom)
        center_offset (Matrix3x1): Sigma has no offset (center on bond axis), Pi has +/- offsets for side lobes
        radial_scale (float): Thickness of cloud perpendicular to bond axis
        alpha (float): Transparency level

    Raises:
        RuntimeError: If the ShowBase loader is not initialized

    Returns:
        NodePath: The electron cloud
    """
    if base.loader is None:
        raise RuntimeError("ShowBase loader is not initialized")

    cloud = cast(NodePath, base.loader.loadModel("models/misc/sphere"))
    cloud.reparentTo(parent)

    mid = ((start + end) * 0.5) + center_offset
    length = float(np.linalg.norm(end - start))
    if length < 1e-8:
        length = 1e-8

    cloud.setPos(float(mid[0]), float(mid[1]), float(mid[2]))
    cloud.lookAt(float(end), float(end), float(end[2]))
    cloud.setScale(radial_scale, length * 0.5, radial_scale)

    cloud.setTransparency(TransparencyAttrib.MAlpha)
    cloud.setColor(color[0], color[1], color[2], alpha)


def _perpendicular_axes(unit_axis: Matrix3x1) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes the bond axis direction vector, picks a reference vector that is not parallel, builds a
    vector (u) to be perpendicular to bond axis, and another (v) perpendicular to bond axis and u

    Args:
        unit_axis (Matrix3x1): The vector representing the bond axis

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing both perpendicular vectors.
    """
    ref = np.asarray([0.0, 0.0, 1.0])
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
        base (ShowBase): Panda3D Base app object.
        parent (NodePath): Node parent.
        atom_a (np.ndarray): Coordinates of first atom in local molecule space.
        atom_b (np.ndarray): Coordinates of second atom in local molecule space.
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
        base (ShowBase): Panda3D Base app object.
        parent (NodePath): Node parent.
        atom_a (np.ndarray): Coordinates of first atom in local molecule space.
        atom_b (np.ndarray): Coordinates of second atom in local molecule space.
        offset_axis (np.ndarray): Unit axis perpendicular to the bond axis used for lobe offsets.

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
    Creates all bond cloud nodes for a bond using sigma and pi components.

    Args:
        base (ShowBase): Panda3D Base app object.
        parent (NodePath): Node parent.
        atom_a (np.ndarray): Coordinates of first atom in local molecule space.
        atom_b (np.ndarray): Coordinates of second atom in local molecule space.
        bond_order (int): Bond order used to determine sigma/pi components.

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

    return visuals
