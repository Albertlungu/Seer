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
        )
        self.environ.reparentTo(self.render)
        self.environ.setP(90)
        self.environ.setTwoSided(True)

        center = self.environ.getTightBounds()
        mid = (center[0] + center[1]) / 2
        self.environ.setPos(-mid.x, -mid.y, -mid.z)

        self.camLens.setNear(0.01)
        self.camLens.setFov(90)

        # Putting the mouse to the center of the window, locking it, and making it hidden
        props = WindowProperties()
        props.setMouseMode(WindowProperties.M_relative)
        props.setCursorHidden(True)
        base.win.requestProperties(props)


app = Room()
app.run()
