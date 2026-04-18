"""
./src/zoom/raycast_picker.py

python -m src.zoom.raycast_picker

Center-screen raycasting using Panda3D's collision system.
"""

from panda3d.core import (
    BitMask32,
    CollisionHandlerQueue,
    CollisionNode,
    CollisionRay,
    CollisionTraverser,
    NodePath,
    Point2,
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
        camera: NodePath,
        cam_node,
        target_root: NodePath,
        collision_mask: BitMask32 = BitMask32.bit(1),
    ) -> None:
        self.target_root: NodePath = target_root
        self.cam_node = cam_node

        self.traverser: CollisionTraverser = CollisionTraverser()
        self.queue: CollisionHandlerQueue = CollisionHandlerQueue()

        cn: CollisionNode = CollisionNode("center_picker.py")
        cn.setFromCollideMask(collision_mask)
        self.ray_np: NodePath = camera.attachNewNode(cn)

        self.ray: CollisionRay = CollisionRay()
        cn.addSolid(self.ray)
        self.traverser.addCollider(self.ray_np, self.queue)

        def pick(self) -> tuple[str, Point3] | None:
            """
            Fire the ray from screen centre and return the first hit.

            Returns:
                tuple[str, Point3] | None: Tuple of (node_name, hit_point_world) or None if nothing was hit.
            """
            self.ray.setFromLens(self.cam_node, Point2(0, 0))
            self.traverser.traverse(self.target_root)

            if self.queue.getNumEntries() == 0:
                return None

            self.queue.sortEntries()
            entry = self.queue.getEntry(0)
            node_name: str = entry.getIntoNode().getName()
            hit_point: Point3 = entry.getSurfacePoint(self.target_root)

            return node_name, hit_point

        def destroy(self) -> None:
            """
            Remove the collision ray from the scene graph.
            """
            self.ray_np.removeNode()
