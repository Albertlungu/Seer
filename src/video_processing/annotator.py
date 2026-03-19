"""
./src/video_processing/annotator.py

This is a manual annotator where the user makes 3D objects themselves as bounding boxes.
Press E to enter annotation mode, click anchor, drag base, height, resize with handles, press
space to name and save.
"""

from __future__ import annotations

from enum import Enum, auto

import open3d as o3d
from direct.showbase.ShowBase import ShowBase
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


class BoxAnnotator(ShowBase):
    def __init__(self):
        super().__init__()
        self.disableMouse()

        self.environ = self.loader.loadModel(OBJ_PATH)
        self.environ.reparentTo(self.render)
        self.environ.setP(90)
        self.environ.setTwoSided(True)
        bounds = self.environ.getTightBounds()
        mid = (bounds[0] + bounds[1]) / 2  # Setting the middle of the scene
        self.environ.setPos(-mid.x, -mid.y, -mid.z)

        mesh = o3d.io.read_triangle_mesh(OBJ_PATH)
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

        self.o3d_scene = o3d.t.geometry.RaycastingScene()
        self.o3d_scene.add_triangles(mesh_t)

        self.camLens.setNear(0.01)
        self.camLens.setFov(90)  # Will be adjustable

        # Movement
        self.sensitivity = 0.1
        self.heading = 0
        self.pitch = 0
        self.mouse_locked = True
        self.move_speed = 1.5
        self.delta_speed = 0.5
        # Setting the keys to false as default
        self.keys = {"w": False, "s": False, "a": False, "d": False}

        props = WindowProperties()
        props.setCursorHidden(True)  # Hiding the cursor
        self.win.requestProperties(props)

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

        # Making the annotations states
        self.state = AnnotatorState.NAVIGATING  # Default navigation
        self.color_idx = 0
        self.annotations = {}  # Storing them in a dict (will go to JSON)
        self.dialog = None

        # Making the box geometry states, which reset between annotations
        self.anchor = (
            None  # The point that is clicked first on the mesh, acting as an anchor
        )
        self.normal = None  # The vector that points perpendicular to a surface (not in the plane of the surface)
        self.T1 = None  # The vector tangent on the plane of the surface
        self.T2 = None  # The vector perpendicular to T1 on the same plane
        self.dx = 0.0  # Distance along T1 (width)
        self.dy = 0.0  # Distance along T2 (length)
        self.height = 0.0
        self.dragging_handle = None
        self.handle_nodes = []
        self.preview_node = self.environ.attachNewNode("preview")

        # Making the collision picker for the handle dragging
        self.picker = CollisionTraverser()
        self.pick_queue = CollisionHandlerQueue()
        pick_cn = CollisonNode("picker_ray")
        self.pick_np = self.camera.attachNewNode(pick_cn)

        pick_cn.setFromCollideMask(BitMask32.bit(1))
        self.pick_ray = CollisionRay()
        pick_cn.addSolid(self.pick_ray)
        self.picker.addCollider(self.pick_np, self.pick_queue)

        taskMgr.add(self.mouse_look, "mouse-look")
        taskMgr.add(self.move, "move")
        taskMgr.add(self.preview, "preview")

        self.accept("escape", self.toggle_mouse_lock)
        self.accept("shift-=", self.control_speed, [self.delta_speed])
        self.accept("-", self.control_speed, [-self.delta_speed])
        self.accept("wheel_up", self.zoom, [-2])
        self.accept("wheel_down", self.zoom, [2])
        for key in self.keys:
            self.accept(key, self.set_key, [key, True])
            self.accept(f"{key}-up", self.set_key, [key, False])

        self.accept("e", self.toggle_annotate)
        self.accept("mouse1", self.on_click)
        self.accept("mouse1-up", self.on_click_release)
        self.accept("space", self.on_space)
        self.accept("tab", self.on_tab)
        self.accept("s", self.save)

    def toggle_mouse_lock(self):
        pass

    def toggle_annotate(self):
        pass

    def control_speed(self, delta):
        pass

    def zoom(self, delta):
        pass

    def set_key(self, key, value):
        pass

    def on_click(self):
        pass

    def on_click_release(self):
        pass

    def on_space(self):
        pass

    def on_tab(self):
        pass

    def save(self):
        pass
