"""
./src/video_processing/annotator.py

python -m src.video_processing.material_tagging.annotator

This is a manual annotator where the user makes 3D objects themselves as bounding boxes.
Press E to enter annotation mode, click anchor, drag base, height, resize with handles, press
space to name and save.
"""

from __future__ import annotations

import json
import os
from enum import Enum, auto
from typing import Any, cast

import numpy as np
import open3d as o3d
from direct.gui.DirectGui import DirectButton, DirectEntry, DirectFrame, DirectLabel
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBaseGlobal import globalClock
from src.video_processing.environment import Room
from numpy.typing import NDArray
from panda3d.core import (
    BitMask32,
    CardMaker,
    CollisionHandlerQueue,
    CollisionNode,
    CollisionRay,
    CollisionSphere,
    CollisionTraverser,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LineSegs,
    Point2,
    PythonTask,
    TextNode,
    TransparencyAttrib,
    WindowProperties,
    loadPrcFileData,
)

loadPrcFileData("", "load-file-type p3assimp")

OBJ_PATH = "./data/reconstructions/obj/albert_room.obj"
OUTPUT_PATH = "./data/vision_json/annotations.json"

COLOURS: list[tuple[float, float, float]] = [
    (1.0, 0.2, 0.2),
    (0.2, 1.0, 0.2),
    (0.2, 0.6, 1.0),
    (1.0, 1.0, 0.2),
    (1.0, 0.5, 0.1),
    (0.8, 0.2, 1.0),
    (0.2, 1.0, 0.9),
    (1.0, 0.2, 0.8),
]

BOX_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]

FACE_INDICES: list[list[int]] = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [3, 2, 6, 7],
    [1, 2, 6, 5],
    [0, 3, 7, 4],
]


# Lookup dictionary used to find the corner opposite diagonally
# It's so that when I resize from one corner, the other can stay fixed
OPPOSITE: dict[int, int] = {
    0: 2,
    1: 3,
    2: 0,
    3: 1,
}


# State machine using enums
class AnnotatorState(Enum):
    """
    State machine for annotation flow.
    """

    NAVIGATING = auto()
    IDLE = auto()
    DRAWING = auto()  # The user is controlling width and height
    HEIGHT = auto()  # If active; the user is actively editing the box's height
    EDITING = auto()
    DIALOG = (
        auto()
    )  # The save form (when the user presses the save key, a dialog opens)


# Allowing the each annotator state to have multiple complementary states, so you can be in
# drawing mode but idling
ALLOWED_TRANSITIONS: dict[AnnotatorState, set[AnnotatorState]] = {
    AnnotatorState.NAVIGATING: {AnnotatorState.IDLE},
    AnnotatorState.IDLE: {AnnotatorState.NAVIGATING, AnnotatorState.DRAWING},
    AnnotatorState.DRAWING: {AnnotatorState.HEIGHT, AnnotatorState.IDLE},
    AnnotatorState.HEIGHT: {AnnotatorState.EDITING, AnnotatorState.IDLE},
    AnnotatorState.EDITING: {AnnotatorState.DIALOG, AnnotatorState.IDLE},
    AnnotatorState.DIALOG: {AnnotatorState.EDITING, AnnotatorState.IDLE},
}


class BoxAnnotator(Room):
    """
    TinkerCAD-style 3D bbox annotation maker.

    Inherits camera navigation and movement from Room.
    """

    def __init__(self) -> None:
        """
        Initializing the annotator by creating the environment.
        """
        # Initialize Room (parent class) - sets up environment, camera, input handling
        super().__init__()

        # Casting dynamic Panda3D attributes to Any so static analysis can resolve method calls
        self.environ = cast(Any, self.environ)
        self.camera = cast(Any, self.camera)
        self.camNode = cast(Any, self.camNode)
        self.camLens = cast(Any, self.camLens)
        self.win = cast(Any, self.win)
        self.loader = cast(Any, self.loader)
        self.mouseWatcherNode = cast(Any, self.mouseWatcherNode)

        mesh = o3d.io.read_triangle_mesh(OBJ_PATH)
        mesh.compute_vertex_normals()
        self.triangles: NDArray[np.int32] = np.asarray(mesh.triangles, dtype=np.int32)
        self.vertex_normals: NDArray[Any] = np.asarray(
            mesh.vertex_normals, dtype=np.float32
        )
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self.o3d_scene = o3d.t.geometry.RaycastingScene()
        self.o3d_scene.add_triangles(mesh_t)

        # Drawing the actual "+" crosshair
        # Units: normalized, [1, 1]
        ls: LineSegs = LineSegs()  # Creates a line builder in Panda3D
        ls.setColor(1, 1, 1, 1)  # White color
        ls.setThickness(1.5)
        ls.moveTo(-0.02, 0, 0)  # Draws the horizontal line
        ls.drawTo(0.02, 0, 0)
        ls.moveTo(0, 0, -0.02)  # Draws the vertical line
        ls.drawTo(0, 0, 0.02)
        self.render2d.attachNewNode(ls.create())  # Renders it

        # Annotation state machine
        self.state: AnnotatorState = AnnotatorState.NAVIGATING  # Default navigation
        self.color_idx: int = 0
        self.annotations: dict[str, dict[str, object]] = {}  # Will be saved to JSON
        self.saved_boxes: dict[str, Any] = {}  # key → persistent scene node
        self.saved_x_markers: dict[str, Any] = {}  # key → X button node
        self.dialog: DirectFrame | None = None

        # Status text in top-right corner (shows current state)
        self.status_text: OnscreenText = OnscreenText(
            text="",
            parent=self.aspect2d,
            pos=(1.28, 0.93),
            scale=0.05,
            fg=(1, 1, 1, 1),
            align=TextNode.ARight,
            mayChange=True,
        )
        self._update_status_text()

        # Ray debug visuals so user can see cast direction and hit point
        self.show_rays: bool = True
        self.ray_debug_node = self.environ.attachNewNode("ray_debug")
        self.ray_hit_marker = self.loader.loadModel("models/misc/sphere")
        self.ray_hit_marker.setScale(0.015)
        self.ray_hit_marker.setColor(0.1, 1.0, 0.1, 1.0)
        self.ray_hit_marker.reparentTo(self.environ)
        self.ray_hit_marker.hide()

        # Making the box geometry states, which reset between annotations
        self.anchor: NDArray[Any] | None = (
            None  # The point that is clicked first on the mesh, acting as an anchor
        )
        self.normal: NDArray[Any] | None = (
            None  # The vector that points perpendicular to a surface
        )
        self.T1: NDArray[Any] | None = (
            None  # The vector tangent on the plane of the surface
        )
        self.T2: NDArray[Any] | None = (
            None  # The vector perpendicular to T1 on the same plane
        )
        self.dx: float = 0.0  # Distance along T1 (width)
        self.dy: float = 0.0  # Distance along T2 (length)
        self.height: float = 0.0
        self.dragging_handle: int | None = None
        self.handle_nodes: list[Any] = []
        self.preview_node = self.environ.attachNewNode("preview")

        # Making the collision picker for the handle dragging
        self.picker: CollisionTraverser = CollisionTraverser()
        self.pick_queue: CollisionHandlerQueue = CollisionHandlerQueue()
        pick_cn: CollisionNode = CollisionNode("picker_ray")
        self.pick_np = self.camera.attachNewNode(pick_cn)

        pick_cn.setFromCollideMask(BitMask32.bit(1) | BitMask32.bit(2))
        self.pick_ray: CollisionRay = CollisionRay()
        pick_cn.addSolid(self.pick_ray)
        self.picker.addCollider(self.pick_np, self.pick_queue)

        # Register preview task (runs every frame while app is running)
        self.taskMgr.add(self.preview, "preview")

        # Annotation controls (kept separate from inherited movement controls)
        self.accept("e", self.toggle_annotate)
        self.accept("mouse1", self.on_click)
        self.accept("mouse1-up", self.on_click_release)
        self.accept("space", self.on_space)
        self.accept("tab", self.on_tab)
        self.accept("control-s", self.save)
        self.accept("r", self.toggle_raycast_visual)

    def _set_state(self, new_state: AnnotatorState) -> None:
        """
        Changes states.

        Args:
            new_state (AnnotatorState): The new state to be changed to.

        Raises:
            RuntimeError: If the transition is invalid (incorrect resultant state)
        """
        if new_state == self.state:
            return

        # Guarding state transitions so handlers cannot jump to invalid state
        if new_state not in ALLOWED_TRANSITIONS[self.state]:
            raise RuntimeError(f"Invalid transition: {self.state} -> {new_state}")
        self.state = new_state
        self._update_status_text()

    def _update_status_text(self) -> None:
        """
        Updates top-right status indicator based on current state.
        """
        self.status_text.setText(f"{self.state.name}")

    def _clear_movement_keys(self) -> None:
        """
        Clears WASD key state to prevent stuck movement across mode switches.
        """
        self.keys["w"] = False
        self.keys["a"] = False
        self.keys["s"] = False
        self.keys["d"] = False

    def _center_mouse_pointer(self) -> None:
        """
        Recenters OS pointer in the window for consistent center-ray behaviour.
        """
        width: int = self.win.getXSize()
        height: int = self.win.getYSize()
        if width <= 0 or height <= 0:
            return

        self.win.movePointer(0, width // 2, height // 2)

    def toggle_raycast_visual(self) -> None:
        """
        Toggles raycast visualization on and off.
        """
        self.show_rays = not self.show_rays
        if not self.show_rays:
            self.ray_debug_node.node().removeAllChildren()
            self.ray_hit_marker.hide()

    def toggle_annotate(self) -> None:
        """
        Toggles between navigation and annotation modes.

        If called in a drafting/editing state, it safely exits annotation mode,
        discarding in-progress geometry and returning to NAVIGATING.
        """
        if self.dialog is not None:
            return

        if self.state == AnnotatorState.NAVIGATING:
            self._clear_movement_keys()
            self._set_state(AnnotatorState.IDLE)
            self.mouse_locked = False
            props: WindowProperties = WindowProperties()
            props.setCursorHidden(False)
            self.win.requestProperties(props)

        elif self.state == AnnotatorState.IDLE:
            self._clear_movement_keys()
            self._set_state(AnnotatorState.NAVIGATING)
            self.mouse_locked = True
            props: WindowProperties = WindowProperties()
            props.setCursorHidden(True)
            self.win.requestProperties(props)
            self._center_mouse_pointer()

        elif self.state == AnnotatorState.DIALOG:
            # Close dialog without discarding the in-progress box, same as Cancel
            self.cancel_dialog()

        else:
            # DRAWING / HEIGHT / EDITING
            self._reset()
            self._clear_movement_keys()
            self._set_state(AnnotatorState.NAVIGATING)
            self.mouse_locked = True
            props: WindowProperties = WindowProperties()
            props.setCursorHidden(True)
            self.win.requestProperties(props)
            self._center_mouse_pointer()

    def toggle_mouse_lock(self) -> None:
        """
        Handles Escape key safely across all annotator states.

        - NAVIGATING: toggle mouse lock
        - DIALOG: close dialog and return to editing
        - Any annotation state: exit annotation mode and return to NAVIGATING
        """
        if self.state == AnnotatorState.NAVIGATING:
            self.mouse_locked = not self.mouse_locked
            props: WindowProperties = WindowProperties()
            props.setCursorHidden(self.mouse_locked)
            self.win.requestProperties(props)
            if self.mouse_locked:
                self._center_mouse_pointer()
            return

        if self.state == AnnotatorState.DIALOG:
            self.cancel_dialog()
            return

        # IDLE / DRAWING / HEIGHT / EDITING -> exit to navigation safely
        if self.state in {
            AnnotatorState.IDLE,
            AnnotatorState.DRAWING,
            AnnotatorState.HEIGHT,
            AnnotatorState.EDITING,
        }:
            self._reset()
            self._clear_movement_keys()
            self._set_state(AnnotatorState.NAVIGATING)
            self.mouse_locked = True
            props: WindowProperties = WindowProperties()
            props.setCursorHidden(True)
            self.win.requestProperties(props)
            self._center_mouse_pointer()

    def move(self, task: PythonTask) -> PythonTask:
        """
        Moves camera only in NAVIGATING mode.

        This prevents WASD from moving camera while the user is annotating.

        Args:
            task (PythonTask): Panda3D task object.

        Returns:
            PythonTask: task.cont to keep task running.
        """
        if self.state == AnnotatorState.DIALOG:
            return task.cont
        if self.state == AnnotatorState.NAVIGATING and not self.mouse_locked:
            return task.cont

        dt: float = globalClock.getDt()
        if self.keys["w"]:
            self.camera.setY(self.camera, self.move_speed * dt)
        if self.keys["s"]:
            self.camera.setY(self.camera, -self.move_speed * dt)
        if self.keys["a"]:
            self.camera.setX(self.camera, -self.move_speed * dt)
        if self.keys["d"]:
            self.camera.setX(self.camera, self.move_speed * dt)

        return task.cont

    # --- per-frame preview task ---

    def preview(self, task: PythonTask) -> PythonTask:
        """
        Updates the box preview based on current state and mouse position.

        Args:
            task (Task): Panda3D task object that runs every frame.

        Returns:
            Task: task.cont to keep task running.
        """
        self._update_raycast_visual()

        if self.state == AnnotatorState.DRAWING:
            ray: tuple[NDArray[Any], NDArray[Any]] | None = self._mouse_ray()
            if ray:
                origin: NDArray[Any]
                direction: NDArray[Any]
                origin, direction = ray
                assert self.anchor is not None
                assert self.normal is not None
                assert self.T1 is not None
                assert self.T2 is not None
                hit: NDArray[Any] | None = self._plane_intersect(
                    origin, direction, self.anchor, self.normal
                )
                if hit is not None:
                    # Mapping hit point to tangent frame dimensions (dx, dy)
                    diag: NDArray[Any] = hit - self.anchor
                    self.dx = float(np.dot(diag, self.T1))
                    self.dy = float(np.dot(diag, self.T2))
                    self.height = 0.0
                    self._rebuild_preview()

        elif self.state == AnnotatorState.HEIGHT:
            ray: tuple[NDArray[Any], NDArray[Any]] | None = self._mouse_ray()
            if ray:
                # Height is clamped to avoid zero-height faces
                self.height = max(0.001, self._height_from_ray(*ray))
                self._rebuild_preview()

        elif self.state == AnnotatorState.EDITING and self.dragging_handle is not None:
            ray = self._mouse_ray()
            if ray:
                self._drag_handle(*ray)

        return task.cont

    def _update_raycast_visual(self) -> None:
        """
        Draws current interaction ray and hit location in the scene.

        Green line means mesh hit.
        Red line means no hit.
        """
        if not self.show_rays:
            return

        ray: tuple[NDArray[Any], NDArray[Any]] | None = self._interaction_ray()
        if ray is None:
            return

        origin: NDArray[Any]
        direction: NDArray[Any]
        origin, direction = ray

        ray_t: o3d.core.Tensor = o3d.core.Tensor(
            [[*origin, *direction]], dtype=o3d.core.float32
        )
        res: dict[str, o3d.core.Tensor] = self.o3d_scene.cast_rays(ray_t)
        t_hit: float = float(res["t_hit"].numpy()[0])

        hit_found: bool = not np.isinf(t_hit)
        end_point: NDArray[Any]
        if hit_found:
            end_point = origin + t_hit * direction
            self.ray_hit_marker.setPos(*end_point)
            self.ray_hit_marker.show()
        else:
            end_point = origin + 2.0 * direction
            self.ray_hit_marker.hide()

        self.ray_debug_node.node().removeAllChildren()

        ls: LineSegs = LineSegs()
        if hit_found:
            ls.setColor(0.1, 1.0, 0.1, 1.0)
        else:
            ls.setColor(1.0, 0.2, 0.2, 1.0)
        ls.setThickness(2.5)
        ls.moveTo(*origin)
        ls.drawTo(*end_point)
        self.ray_debug_node.attachNewNode(ls.create())

    # --- input handlers ---

    def on_click(self) -> None:
        """
        Handles mouse click events for all annotation states.
        """
        if self.state == AnnotatorState.IDLE:
            # Check for X marker click before placing a new anchor
            if self.mouseWatcherNode.hasMouse():
                mx: float = self.mouseWatcherNode.getMouseX()
                my: float = self.mouseWatcherNode.getMouseY()
                self.pick_ray.setFromLens(self.camNode, Point2(mx, my))
                self.picker.traverse(self.environ)
                if self.pick_queue.getNumEntries() > 0:
                    self.pick_queue.sortEntries()
                    hit_name: str = self.pick_queue.getEntry(0).getIntoNode().getName()
                    if hit_name.startswith("x_marker_"):
                        self._delete_saved_box(hit_name[len("x_marker_") :])
                        return

            result: tuple[NDArray[Any], NDArray[Any]] | None = self._raycast_mouse()
            if result is None:
                return
            hit: NDArray[Any]
            normal: NDArray[Any]
            hit, normal = result

            # Ensuring normal faces towards camera direction
            cam_origin: NDArray[Any]
            cam_origin, _ = self._ray_obj_space()
            if np.dot(normal, cam_origin - hit) < 0:
                normal = -normal

            # Building tangent frame from normal vector
            # World Z-up is [0, 1, 0] in environ's OBJ space because environ has setP(90).
            # Fallback for horizontal surfaces (floor/ceiling) uses world X = OBJ [1, 0, 0].
            up: NDArray[Any] = (
                np.array([0.0, 1.0, 0.0], dtype=np.float32)
                if abs(normal[1]) < 0.9
                else np.array([1.0, 0.0, 0.0], dtype=np.float32)
            )
            T1: NDArray[Any] = np.cross(normal, up)
            T1 /= np.linalg.norm(T1)
            T2: NDArray[Any] = np.cross(T1, normal)
            T2 /= np.linalg.norm(T2)

            self.anchor = hit
            self.normal = normal
            self.T1 = T1
            self.T2 = T2
            self.dx = 0.0
            self.dy = 0.0
            self.height = 0.0
            self._set_state(AnnotatorState.DRAWING)

        elif self.state == AnnotatorState.DRAWING:
            if abs(self.dx) < 1e-3 or abs(self.dy) < 1e-3:
                return
            self._set_state(AnnotatorState.HEIGHT)

        elif self.state == AnnotatorState.HEIGHT:
            if abs(self.height) < 1e-3:
                return
            self._spawn_handles()
            self._set_state(AnnotatorState.EDITING)

        elif self.state == AnnotatorState.EDITING:
            if not self.mouseWatcherNode.hasMouse():
                return
            mx: float = self.mouseWatcherNode.getMouseX()
            my: float = self.mouseWatcherNode.getMouseY()
            self.pick_ray.setFromLens(self.camNode, Point2(mx, my))
            self.picker.traverse(self.environ)
            if self.pick_queue.getNumEntries() > 0:
                self.pick_queue.sortEntries()
                name: str = self.pick_queue.getEntry(0).getIntoNode().getName()
                if name.startswith("handle_"):
                    # Store corner index so preview task can drag this handle
                    self.dragging_handle = int(name.split("_")[1])

    def on_click_release(self) -> None:
        """
        Stops handle dragging when mouse is released.
        """
        if self.state == AnnotatorState.EDITING:
            self.dragging_handle = None

    def on_space(self) -> None:
        """
        Opens save dialog when in editing state.
        """
        if self.state == AnnotatorState.EDITING and self.dialog is None:
            self._open_dialog()

    def on_tab(self) -> None:
        """
        Resets current in-progress box and advances colour.
        """
        if self.state in {AnnotatorState.NAVIGATING, AnnotatorState.DIALOG}:
            return
        self._reset()
        self.color_idx += 1

    # --- raycasting ---

    def _ray_obj_space(
        self, screen_pos: Point2 | None = None
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """
        Raycasts in environment's local object space.

        Args:
            screen_pos (Point2 | None): Normalized screen position. Defaults to center.

        Returns:
            tuple[NDArray[Any], NDArray[Any]]: Ray origin and direction.
        """
        screen: Point2 = Point2(0, 0) if screen_pos is None else screen_pos

        sx: float = float(screen.getX())
        sy: float = float(screen.getY())

        # Use Panda's own lens ray construction to match camera controls exactly.
        picker_ray: CollisionRay = CollisionRay()
        picker_ray.setFromLens(self.camNode, sx, sy)

        ray_o_cam: Any = picker_ray.getOrigin()
        ray_d_cam: Any = picker_ray.getDirection()

        # Convert directly from camera space into environ's local (OBJ) space.
        # environ has setP(90) and setPos(-mid) as world transforms; its local space
        # matches the raw OBJ coordinate system that Open3D loaded — so this
        # conversion gives a ray that aligns with the Open3D raycasting scene.
        o: Any = self.environ.getRelativePoint(self.camera, ray_o_cam)
        d: Any = self.environ.getRelativeVector(self.camera, ray_d_cam)
        d.normalize()

        origin: NDArray[Any] = np.array([o.x, o.y, o.z], dtype=np.float32)
        direction: NDArray[Any] = np.array([d.x, d.y, d.z], dtype=np.float32)
        return origin, direction

    def _mouse_ray(self) -> tuple[NDArray[Any], NDArray[Any]] | None:
        """
        Gets ray origin and direction from current mouse position.

        Returns:
            tuple[NDArray[Any], NDArray[Any]] | None: Ray tuple if mouse is valid.
        """
        if not self.mouseWatcherNode.hasMouse():
            return None

        # Mouse coords are normalized in range [-1, 1]
        mx: float = self.mouseWatcherNode.getMouseX()
        my: float = self.mouseWatcherNode.getMouseY()
        return self._ray_obj_space(Point2(mx, my))

    def _interaction_ray(
        self,
    ) -> tuple[NDArray[Any], NDArray[Any]] | None:
        """
        Gets ray for active interaction mode.

        In mouse-lock mode this always uses screen center.
        """
        if self.mouse_locked:
            return self._ray_obj_space(Point2(0, 0))
        return self._mouse_ray()

    def _interpolate_normal(self, res: dict[str, o3d.core.Tensor]) -> NDArray[Any]:
        tri_idx: int = int(res["primitive_ids"].numpy()[0])
        u: float
        v: float
        u, v = res["primitive_uvs"].numpy()[0]
        w: float = 1.0 - u - v
        tri: NDArray[np.int32] = self.triangles[tri_idx]
        normal: NDArray[Any] = (
            w * self.vertex_normals[tri[0]]
            + u * self.vertex_normals[tri[1]]
            + v * self.vertex_normals[tri[2]]
        )
        normal /= np.linalg.norm(normal)
        return normal

    def _raycast_center(
        self,
    ) -> tuple[NDArray[Any], NDArray[Any]] | None:
        """
        Casts ray from screen center and returns hit and normal.

        Returns:
            tuple[NDArray[Any], NDArray[Any]] | None: Hit point and primitive normal.
        """
        origin: NDArray[Any]
        direction: NDArray[Any]
        origin, direction = self._ray_obj_space()

        ray: o3d.core.Tensor = o3d.core.Tensor(
            [[*origin, *direction]], dtype=o3d.core.float32
        )
        res: dict[str, o3d.core.Tensor] = self.o3d_scene.cast_rays(ray)
        t_hit: float = float(res["t_hit"].numpy()[0])

        # No mesh hit means ray did not intersect geometry
        if np.isinf(t_hit):
            return None

        hit: NDArray[Any] = origin + t_hit * direction
        normal: NDArray[Any] = self._interpolate_normal(res)
        return hit, normal

    def _raycast_mouse(
        self,
    ) -> tuple[NDArray[Any], NDArray[Any]] | None:
        """
        Casts ray from current mouse and returns hit and normal.

        Returns:
            tuple[NDArray[Any], NDArray[Any]] | None: Hit point and primitive normal.
        """
        ray: tuple[NDArray[Any], NDArray[Any]] | None = self._mouse_ray()
        if ray is None:
            return None

        origin: NDArray[Any]
        direction: NDArray[Any]
        origin, direction = ray

        ray_t: o3d.core.Tensor = o3d.core.Tensor(
            [[*origin, *direction]], dtype=o3d.core.float32
        )
        res: dict[str, o3d.core.Tensor] = self.o3d_scene.cast_rays(ray_t)
        t_hit: float = float(res["t_hit"].numpy()[0])
        if np.isinf(t_hit):
            return None

        hit: NDArray[Any] = origin + t_hit * direction
        normal: NDArray[Any] = self._interpolate_normal(res)
        return hit, normal

    def _plane_intersect(
        self,
        ray_o: NDArray[Any],
        ray_d: NDArray[Any],
        plane_p: NDArray[Any],
        plane_n: NDArray[Any],
    ) -> NDArray[Any] | None:
        """
        Calculates ray-plane intersection point.

        Args:
            ray_o (NDArray[Any]): Ray origin.
            ray_d (NDArray[Any]): Ray direction.
            plane_p (NDArray[Any]): Point on plane.
            plane_n (NDArray[Any]): Plane normal.

        Returns:
            NDArray[Any] | None: Intersection point if valid.
        """
        denom: float = float(np.dot(ray_d, plane_n))

        # Parallel ray means no stable intersection with this plane
        if abs(denom) < 1e-6:
            return None

        t: float = float(np.dot(plane_p - ray_o, plane_n) / denom)
        return None if t < 0 else ray_o + t * ray_d

    def _height_from_ray(
        self,
        ray_o: NDArray[Any],
        ray_d: NDArray[Any],
    ) -> float:
        """
        Maps mouse ray to height value along current surface normal.

        Args:
            ray_o (NDArray[Any]): Ray origin.
            ray_d (NDArray[Any]): Ray direction.

        Returns:
            float: Height extrusion amount.
        """
        base_center: NDArray[Any] = (
            cast(NDArray[Any], self.anchor)
            + 0.5 * self.dx * cast(NDArray[Any], self.T1)
            + 0.5 * self.dy * cast(NDArray[Any], self.T2)
        )
        _, cam_dir = self._ray_obj_space()
        cam_right: NDArray[Any] = np.cross(cam_dir, cast(NDArray[Any], self.normal))
        n: float = float(np.linalg.norm(cam_right))
        if n < 1e-6:
            return self.height

        cam_right /= n
        hit: NDArray[Any] | None = self._plane_intersect(
            ray_o, ray_d, base_center, cam_right
        )
        if hit is None:
            return self.height

        # Projecting offset onto surface normal gives signed height value
        return float(np.dot(hit - base_center, cast(NDArray[Any], self.normal)))

    # --- box geometry ---

    def _corners(self) -> NDArray[Any]:
        """
        Computes all 8 box corners from current anchor, tangents, and height.

        Returns:
            NDArray[Any]: Array of shape (8, 3) for bottom and top corners.
        """
        bottom: NDArray[Any] = np.array(
            [
                cast(NDArray[Any], self.anchor),
                cast(NDArray[Any], self.anchor) + self.dx * cast(NDArray[Any], self.T1),
                cast(NDArray[Any], self.anchor)
                + self.dx * cast(NDArray[Any], self.T1)
                + self.dy * cast(NDArray[Any], self.T2),
                cast(NDArray[Any], self.anchor) + self.dy * cast(NDArray[Any], self.T2),
            ],
            dtype=np.float32,
        )
        top: NDArray[Any] = bottom + self.height * cast(NDArray[Any], self.normal)
        return np.vstack([bottom, top])

    def _rebuild_preview(self) -> None:
        """
        Clears and redraws the preview box (wireframe + transparent faces).
        """
        self.preview_node.node().removeAllChildren()
        corners: NDArray[Any] = self._corners()

        r: float
        g: float
        b: float
        r, g, b = COLOURS[self.color_idx % len(COLOURS)]

        ls: LineSegs = LineSegs()
        ls.setColor(r, g, b, 1)
        ls.setThickness(2)
        for a, b_idx in BOX_EDGES:
            ls.moveTo(*corners[a])
            ls.drawTo(*corners[b_idx])
        self.preview_node.attachNewNode(ls.create())

        fmt: GeomVertexFormat = GeomVertexFormat.getV3c4()
        vdata: GeomVertexData = GeomVertexData("f", fmt, Geom.UHStatic)
        vw: GeomVertexWriter = GeomVertexWriter(vdata, "vertex")
        cw: GeomVertexWriter = GeomVertexWriter(vdata, "color")
        for face in FACE_INDICES:
            for idx in face:
                vw.addData3(*corners[idx])
                cw.addData4(r, g, b, 0.10)

        tris: GeomTriangles = GeomTriangles(Geom.UHStatic)
        for i in range(len(FACE_INDICES)):
            base: int = i * 4
            tris.addVertices(base, base + 1, base + 2)
            tris.addVertices(base, base + 2, base + 3)

        geom: Geom = Geom(vdata)
        geom.addPrimitive(tris)
        gn: GeomNode = GeomNode("faces")
        gn.addGeom(geom)
        face_np: Any = self.preview_node.attachNewNode(gn)
        face_np.setTransparency(TransparencyAttrib.MAlpha)
        face_np.setTwoSided(True)
        face_np.setDepthWrite(False)

    def _spawn_handles(self) -> None:
        """
        Spawns a draggable collision sphere at each corner.
        """
        for h in self.handle_nodes:
            h.removeNode()
        self.handle_nodes.clear()

        corners: NDArray[Any] = self._corners()
        r: float
        g: float
        b: float
        r, g, b = COLOURS[self.color_idx % len(COLOURS)]
        s: float = 0.025  # half-size in world units
        for i, corner in enumerate(corners):
            cm: CardMaker = CardMaker(f"handle_{i}_card")
            cm.setFrame(-s, s, -s, s)
            handle_np: Any = self.environ.attachNewNode(cm.generate())
            handle_np.setPos(*corner)
            handle_np.setColor(r, g, b, 1)
            handle_np.setBillboardPointEye()
            cn: CollisionNode = CollisionNode(f"handle_{i}")
            cn.addSolid(CollisionSphere(0, 0, 0, s))
            cn.setIntoCollideMask(BitMask32.bit(1))
            handle_np.attachNewNode(cn)
            self.handle_nodes.append(handle_np)

    def _drag_handle(
        self,
        ray_o: NDArray[Any],
        ray_d: NDArray[Any],
    ) -> None:
        """
        Updates box geometry while dragging a selected handle.

        Args:
            ray_o (NDArray[Any]): Current ray origin.
            ray_d (NDArray[Any]): Current ray direction.
        """
        assert self.dragging_handle is not None
        idx: int = self.dragging_handle
        if idx >= 4:
            # Top handle modifies only extrusion height
            self.height = max(0.001, self._height_from_ray(ray_o, ray_d))
        else:
            # Bottom handle re-anchors against opposite corner and updates dx/dy
            assert self.anchor is not None
            assert self.normal is not None
            assert self.T1 is not None
            assert self.T2 is not None
            bottom: list[NDArray[Any]] = [
                self.anchor,
                self.anchor + self.dx * self.T1,
                self.anchor + self.dx * self.T1 + self.dy * self.T2,
                self.anchor + self.dy * self.T2,
            ]
            fixed: NDArray[Any] = bottom[OPPOSITE[idx]]
            hit: NDArray[Any] | None = self._plane_intersect(
                ray_o, ray_d, self.anchor, self.normal
            )
            if hit is None:
                return
            diag: NDArray[Any] = hit - fixed
            self.anchor = fixed
            self.dx = float(np.dot(diag, self.T1))
            self.dy = float(np.dot(diag, self.T2))

        self._rebuild_preview()
        corners: NDArray[Any] = self._corners()
        for i, h in enumerate(self.handle_nodes):
            h.setPos(*corners[i])

    # --- saved box management ---

    def _spawn_saved_box(self, key: str) -> None:
        """
        Creates a persistent wireframe + face visual for a confirmed annotation.

        Also places an X marker above the box for deletion.
        """
        corners: NDArray[Any] = self._corners()
        r: float
        g: float
        b: float
        r, g, b = COLOURS[self.color_idx % len(COLOURS)]

        box_np: Any = self.environ.attachNewNode(f"saved_box_{key}")

        ls: LineSegs = LineSegs()
        ls.setColor(r, g, b, 1)
        ls.setThickness(2)
        for a, b_idx in BOX_EDGES:
            ls.moveTo(*corners[a])
            ls.drawTo(*corners[b_idx])
        box_np.attachNewNode(ls.create())

        fmt: GeomVertexFormat = GeomVertexFormat.getV3c4()
        vdata: GeomVertexData = GeomVertexData("sf", fmt, Geom.UHStatic)
        vw: GeomVertexWriter = GeomVertexWriter(vdata, "vertex")
        cw: GeomVertexWriter = GeomVertexWriter(vdata, "color")
        for face in FACE_INDICES:
            for idx in face:
                vw.addData3(*corners[idx])
                cw.addData4(r, g, b, 0.10)

        tris: GeomTriangles = GeomTriangles(Geom.UHStatic)
        for i in range(len(FACE_INDICES)):
            base: int = i * 4
            tris.addVertices(base, base + 1, base + 2)
            tris.addVertices(base, base + 2, base + 3)

        geom: Geom = Geom(vdata)
        geom.addPrimitive(tris)
        gn: GeomNode = GeomNode("saved_faces")
        gn.addGeom(geom)
        face_np: Any = box_np.attachNewNode(gn)
        face_np.setTransparency(TransparencyAttrib.MAlpha)
        face_np.setTwoSided(True)
        face_np.setDepthWrite(False)

        self.saved_boxes[key] = box_np

        top_center: NDArray[Any] = corners[4:].mean(axis=0)
        assert self.normal is not None
        above: NDArray[Any] = top_center + 0.06 * self.normal
        self._spawn_x_marker(key, above)

    def _spawn_x_marker(self, key: str, pos: NDArray[Any]) -> None:
        """
        Creates a clickable billboard X button above a saved box.
        """
        x_np: Any = self.environ.attachNewNode(f"x_root_{key}")
        x_np.setPos(*pos)
        x_np.setBillboardPointEye()

        tn: TextNode = TextNode(f"x_text_{key}")
        tn.setText("x")
        tn.setAlign(TextNode.ACenter)
        tn.setTextColor(1, 1, 1, 1)
        tn.setCardColor(0.75, 0.1, 0.1, 0.95)
        tn.setCardAsMargin(0.2, 0.2, 0.1, 0.1)
        text_np: Any = x_np.attachNewNode(tn)
        text_np.setScale(0.04)

        cn: CollisionNode = CollisionNode(f"x_marker_{key}")
        cn.addSolid(CollisionSphere(0, 0, 0, 0.05))
        cn.setIntoCollideMask(BitMask32.bit(2))
        x_np.attachNewNode(cn)

        self.saved_x_markers[key] = x_np

    def _delete_saved_box(self, key: str) -> None:
        """
        Removes a saved box and its X marker from the scene and annotations dict.
        """
        if key in self.saved_boxes:
            self.saved_boxes.pop(key).removeNode()
        if key in self.saved_x_markers:
            self.saved_x_markers.pop(key).removeNode()
        self.annotations.pop(key, None)

    # --- dialog ---

    def _open_dialog(self) -> None:
        """
        Opens the save dialog UI.
        """
        self._set_state(AnnotatorState.DIALOG)
        props: WindowProperties = WindowProperties()
        props.setCursorHidden(False)
        self.win.requestProperties(props)

        # Creating modal frame in aspect2d for name/material input
        self.dialog = DirectFrame(
            frameSize=(-0.55, 0.55, -0.28, 0.28),
            frameColor=(0.08, 0.08, 0.08, 0.92),
            parent=self.aspect2d,
        )
        DirectLabel(
            text="Object name:",
            parent=self.dialog,
            pos=(-0.25, 0, 0.17),
            scale=0.055,
            text_fg=(1, 1, 1, 1),
            frameColor=(0, 0, 0, 0),
        )
        self.name_entry: DirectEntry = DirectEntry(
            parent=self.dialog,
            pos=(0.0, 0, 0.15),
            scale=0.055,
            width=9,
            numLines=1,
            focus=1,
        )
        DirectLabel(
            text="Materials (comma-separated):",
            parent=self.dialog,
            pos=(-0.25, 0, 0.04),
            scale=0.045,
            text_fg=(1, 1, 1, 1),
            frameColor=(0, 0, 0, 0),
        )
        self.mat_entry: DirectEntry = DirectEntry(
            parent=self.dialog,
            pos=(0.0, 0, 0.02),
            scale=0.055,
            width=9,
            numLines=1,
        )
        DirectButton(
            text="Save",
            parent=self.dialog,
            pos=(-0.15, 0, -0.18),
            scale=0.065,
            command=self.confirm_dialog,
        )
        DirectButton(
            text="Cancel",
            parent=self.dialog,
            pos=(0.18, 0, -0.18),
            scale=0.065,
            command=self.cancel_dialog,
        )

    def confirm_dialog(self) -> None:
        """
        Saves current box annotation from dialog fields.
        """
        name: str = self.name_entry.get().strip()
        materials: list[str] = [
            m.strip() for m in self.mat_entry.get().split(",") if m.strip()
        ]
        if not name:
            return

        corners: NDArray[Any] = self._corners()
        key: str = name.lower().replace(" ", "_")

        # Avoiding key overwrite by suffixing duplicates
        if key in self.annotations:
            i: int = 2
            candidate: str = f"{key}_{i}"
            while candidate in self.annotations:
                i += 1
                candidate = f"{key}_{i}"
            key = candidate

        assert self.normal is not None

        self.annotations[key] = {
            "name": name,
            "materials": materials,
            "corners": {
                "bottom": corners[:4].tolist(),
                "top": corners[4:].tolist(),
            },
            "base_normal": self.normal.tolist(),
        }

        if self.dialog is not None:
            self.dialog.destroy()
        self.dialog = None

        # Promote draft geometry to a persistent saved box before clearing state
        self._spawn_saved_box(key)
        self._reset()
        self.color_idx += 1

    def cancel_dialog(self) -> None:
        """
        Cancels dialog and returns to editing mode.
        """
        if self.dialog is not None:
            self.dialog.destroy()
        self.dialog = None
        self._set_state(AnnotatorState.EDITING)

    # --- reset + save ---

    def _reset(self) -> None:
        """
        Clears current draft state and goes back to IDLE.
        """
        self.preview_node.node().removeAllChildren()
        h: Any
        for h in self.handle_nodes:
            h.removeNode()
        self.handle_nodes.clear()
        self.anchor = None
        self.normal = None
        self.T1 = None
        self.T2 = None
        self.dx = 0.0
        self.dy = 0.0
        self.height = 0.0
        self.dragging_handle = None
        self._set_state(AnnotatorState.IDLE)

    def save(self) -> None:
        """
        Saves all annotations to OUTPUT_PATH as JSON.
        """
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as file_obj:
            json.dump(self.annotations, file_obj, indent=2)
        print(f"Saved {len(self.annotations)} annotations to {OUTPUT_PATH}")


app = BoxAnnotator()
app.run()
