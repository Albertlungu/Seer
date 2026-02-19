"""
./src/video_processing/environment.py

Creates the 3D environment in which the user can move around.
"""

from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties, loadPrcFileData

loadPrcFileData("", "load-file-type p3assimp")


class Room(ShowBase):
    def __init__(self):
        super().__init__()

        self.disableMouse()

        self.environ = self.loader.loadModel(
            "/Users/albertlungu/Local/GitHub/Seer/data/reconstructions/obj/albert_room.obj"
        )  # Loads the 3D model
        self.environ.reparentTo(self.render)  # The root of the scene (the topmost node)
        self.environ.setP(90)  # Sets the pitch (rotation around x axis) to 90 deg
        self.environ.setTwoSided(True)

        center = self.environ.getTightBounds()  # Returns a tuple of the minimum and maximum corners of the AABB (axis-aligned bounding)
        mid = (center[0] + center[1]) / 2  # Finds the geometric center of the AABB
        self.environ.setPos(-mid.x, -mid.y, -mid.z)  # Moves to origin (middle of room)

        self.camLens.setNear(0.01)  # Renders things that are very close to camera
        self.camLens.setFov(90)

        self.sensitivity = (
            0.3  # Sens in degrees of cam rotation per pixel of mouse movement
        )
        self.heading = 0  # Left/Right rotation
        self.pitch = 0  # Up/Down tilt

        # Hide cursor and center it
        props = WindowProperties()
        props.setCursorHidden(True)
        base.win.requestProperties(props)

        # Register mouse_look as a per-frame task
        taskMgr.add(self.mouse_look, "mouse-look")
        # taskMgr is the task manager
        # It makes a function be called every frame to the main loop
        # Args:
        #   class: function to call
        #   string: name string

        self.mouse_locked = True
        self.accept("escape", self.toggle_mouse_lock)

    def toggle_mouse_lock(self):
        """
        Toggle mouse lock on/off with Escape.
        """
        self.mouse_locked = not self.mouse_locked
        props = WindowProperties()
        props.setCursorHidden(self.mouse_locked)
        base.win.requestProperties(props)

    def mouse_look(self, task):
        """
        The method to calculate the location of the actual camera.

        Args:
            task (Task): Task object automatically passed by taskMgr. Provides frame timing.
        """
        if (
            self.mouse_locked and self.mouseWatcherNode.hasMouse()
        ):  # Check if mouse is in the window
            # Get window center in pixels
            cx = self.win.getXSize() // 2
            cy = self.win.getYSize() // 2

            # Current mouse position (normalized coordinates)
            nx = self.mouseWatcherNode.getMouseX()
            ny = self.mouseWatcherNode.getMouseY()
            # Current mouse position in pixels (converts normalized)
            px = round((nx + 1) * 0.5 * self.win.getXSize())
            py = round((1 - (ny + 1) * 0.5) * self.win.getYSize())

            # Distance from center
            dx = px - cx
            dy = cy - py  # Reversed for natural mouse movement

            # Dead zone to ignore sub-pixel noise
            if abs(dx) > 1 or abs(dy) > 1:
                self.heading -= dx * self.sensitivity
                self.pitch = max(
                    -80, min(80, self.pitch + dy * self.sensitivity)
                )  # Clamps the vertical degrees to 80 degrees horizontally and vertically
                self.camera.setHpr(
                    self.heading, self.pitch, 0
                )  # Applies the rotation to the camera

                # Makes sure the cursor is always centered at the middle of the screen
                self.win.movePointer(0, cx, cy)
                # After each frame, move the mouse back to center
        return task.cont


app = Room()
app.run()
