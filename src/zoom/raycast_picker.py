"""
./src/zoom/raycast_picker.py

python -m src.zoom.raycast_picker

Center-screen raycasting using Panda3D's collision system.
"""

from panda3d.core import (
    BitMask32,
    Camera,
    CollisionHandlerQueue,
    CollisionNode,
    CollisionRay,
    CollisionTraverser,
    NodePath,
    Point3,
)


class RaycastPicker:
    """
    Fires a collision ray from screen center and reports the first hit.

    Args:
        camera: The camera NodePath.
        cam_node: The Camera node (base.camNode).
        target_root: Scene-graph node to analyze for hits.
        collision_mask: BitMask32 to match against. Defaults to hit 1.
    """

    def __init__(
        self,
        camera_node: NodePath,
        cam_node: Camera,
        target_root: NodePath,
        collision_mask: BitMask32 = BitMask32.bit(1),
    ) -> None:
        self.target_root: NodePath = target_root
        self.camera_node = camera_node
        self.cam_node = cam_node
        self.collision_mask = collision_mask

        self.traverser: CollisionTraverser = CollisionTraverser()
        self.queue: CollisionHandlerQueue = CollisionHandlerQueue()

        cn: CollisionNode = CollisionNode("center_picker.py")
        cn.setFromCollideMask(collision_mask)
        self.ray_np: NodePath = camera_node.attachNewNode(cn)

        self.ray: CollisionRay = CollisionRay()
        cn.addSolid(self.ray)
        self.traverser.addCollider(self.ray_np, self.queue)

    def mark_pickable(self, root: NodePath) -> None:
        """
        Marks GeomNode descendants as pickable by the center ray.
        """
        candidates = [root]
        matches = root.findAllMatches("**/+GeomNode")
        for index in range(matches.getNumPaths()):
            candidates.append(matches.getPath(index))

        for node_path in candidates:
            node = node_path.node()
            set_into_mask = getattr(node, "setIntoCollideMask", None)
            if set_into_mask is not None:
                set_into_mask(self.collision_mask)

    def pick_center(self) -> tuple[NodePath, Point3] | None:
        """
        Fire the ray from screen centre and return the first hit.

        Returns:
            tuple[NodePath, Point3] | None: Tuple of (hit_node_path, hit_point_local_to_target_root) or None if nothing was hit.
        """
        self.queue.clearEntries()
        self.ray.setFromLens(self.cam_node, 0.0, 0.0)
        self.traverser.traverse(self.target_root)

        if self.queue.getNumEntries() == 0:
            return None

        self.queue.sortEntries()
        entry = self.queue.getEntry(0)
        return entry.getIntoNodePath(), entry.getSurfacePoint(self.target_root)

    def destroy(self) -> None:
        """
        Remove the collision ray from the scene graph.
        """
        self.ray_np.removeNode()
