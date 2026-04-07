"""
./src/render_molecules/arrangement/renderer.py

python -m src.render_molecules.arrangement.renderer

Turns arranged molecular states into visible structures. Atoms as normal spheres, but bonds not
as cylinders, but as electron-density regions, reflecting bond order (taking into account sigma/pi bonds)
"""

from typing import cast

import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Mat4, NodePath

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
    for aid in template.aids:
        create_atom_sphere(base=base, parent=root, template=template, aid=int(aid))

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
