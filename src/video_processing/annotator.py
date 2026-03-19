"""
./src/video_processing/annotator.py

This is a manual annotator where the user makes 3D objects themselves as bounding boxes.
Press E to enter annotation mode, click anchor, drag base, height, resize with handles, press
space to name and save.
"""

from __future__ import annotations

from enum import Enum, auto

import open3d as o3d
from environment import Room
from panda3d.core import (
    BitMask32,
    CollisionHandlerQueue,
    CollisionRay,
    CollisionTraverser,
    LineSegs,
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
OPPOSITE = {
    0: 2,
    1: 3,
    2: 0,
    3: 1,
}


# State machine using enums
class AnnotatorState(Enum):
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
    def __init__(self):
        # Initialize Room (parent class) - sets up environment, camera, input handling
        super().__init__()

        # Set up Open3D raycasting for mesh intersections
        mesh = o3d.io.read_triangle_mesh(OBJ_PATH)
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self.o3d_scene = o3d.t.geometry.RaycastingScene()
        self.o3d_scene.add_triangles(mesh_t)

        # Drawing the actual "+" crosshair
        # Units: normalized, [1, 1]
        ls = LineSegs()  # Creates a line builder in Panda3D
        ls.setColor(1, 1, 1, 1)  # White color
        ls.setThickness(1.5)
        ls.moveTo(-0.02, 0, 0)  # Draws the horizontal line
        ls.drawTo(0.02, 0, 0)
        ls.moveTo(0, 0, -0.02)  # Draws the vertical line
        ls.drawTo(0, 0, 0.02)
        self.render2d.attachNewNode(ls.create())  # Renders it

        # Annotation state machine
        self.state = AnnotatorState.NAVIGATING  # Default navigation
        self.color_idx = 0
        self.annotations = {}  # Storing them in a dict (will go to JSON)
        self.dialog = None

        # Box geometry state (reset between annotations)
        self.anchor = None  # The first clicked point on the mesh
        self.normal = None  # Surface normal at anchor
        self.T1 = None  # First tangent vector on surface plane
        self.T2 = None  # Second tangent vector perpendicular to T1
        self.dx = 0.0  # Distance along T1 (width)
        self.dy = 0.0  # Distance along T2 (length)
        self.height = 0.0
        self.dragging_handle = None
        self.handle_nodes = []
        self.preview_node = self.environ.attachNewNode("preview")

        # Collision picker for handle dragging
        self.picker = CollisionTraverser()
        self.pick_queue = CollisionHandlerQueue()
        pick_cn = CollisionNode("picker_ray")
        self.pick_np = self.camera.attachNewNode(pick_cn)
        pick_cn.setFromCollideMask(BitMask32.bit(1))
        self.pick_ray = CollisionRay()
        pick_cn.addSolid(self.pick_ray)
        self.picker.addCollider(self.pick_np, self.pick_queue)

        # Register annotation-specific tasks and input handlers
        taskMgr.add(self.preview, "preview")

        self.accept("e", self.toggle_annotate)
        self.accept("mouse1", self.on_click)
        self.accept("mouse1-up", self.on_click_release)
        self.accept("space", self.on_space)
        self.accept("tab", self.on_tab)
        self.accept("s", self.save)

    def _set_state(self, new_state: AnnotatorState):
        """
        Changes states.

        Args:
            new_state (AnnotatorState): The new state to be changed to.

        Raises:
            RuntimeError: If the transition is invalid (incorrect resultant state)
        """
        if new_state == self.state:
            return
        if new_state not in ALLOWED_TRANSITIONS[self.state]:
            raise RuntimeError(f"Invalid transition: {self.state} -> {new_state}")
        self.state = new_state

    def toggle_annotate(self):
        """
        Toggles between navigation and idle modes.
        """
        if self.state == AnnotatorState.NAVIGATING:
            self._set_state(AnnotatorState.IDLE)
            self.mouse_locked = False
            props = WindowProperties()
            props.setCursorHidden(False)
            self.win.requestProperties(props)
        elif self.state == AnnotatorState.IDLE:
            self._set_state(AnnotatorState.NAVIGATING)
            self.mouse_locked = True
            props = WindowProperties()
            props.setCursorHidden(True)
            self.win.requestProperties(props)

    # --- per-frame preview task ---

    def preview(self, task):
        pass

    # --- input handlers ---

    def on_click(self):
        pass

    def on_click_release(self):
        pass

    def on_space(self):
        pass

    def on_tab(self):
        pass

    # --- raycasting ---

    def _ray_obj_space(self, screen_pos=None):
        pass

    def _mouse_ray(self):
        pass

    def _raycast_center(self):
        pass

    def _raycast_mouse(self):
        pass

    def _plane_intersect(self, ray_o, ray_d, plane_p, plane_n):
        pass

    def _height_from_ray(self, ray_o, ray_d):
        pass

    # --- box geometry ---

    def _corners(self):
        pass

    def _rebuild_preview(self):
        pass

    def _spawn_handles(self):
        pass

    def _drag_handle(self, ray_o, ray_d):
        pass

    # --- dialog ---

    def _open_dialog(self):
        pass

    def confirm_dialog(self):
        pass

    def cancel_dialog(self):
        pass

    # --- reset + save ---

    def _reset(self):
        pass

    def save(self):
        pass
