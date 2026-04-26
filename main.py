"""
Seer application entry point.

Run from source:  python main.py
Bundled:          Seer.exe / Seer.app / Seer (Linux)
"""

import os
import sys

# When bundled with PyInstaller, configure Panda3D paths before ShowBase starts.
if getattr(sys, "frozen", False):
    from panda3d.core import loadPrcFileData

    base = sys._MEIPASS  # type: ignore[attr-defined]
    pd3_dir = os.path.join(base, "panda3d")

    loadPrcFileData("", "\n".join([
        # Tell Panda3D where to find the display plugin (libpandagl.dylib etc.)
        f"plugin-path {pd3_dir}",
        f"plugin-path {base}",
        # Load the OpenGL display backend
        "load-display pandagl",
        # model-path must be the parent of "models/", so loadModel("models/misc/sphere") resolves correctly
        f"model-path {pd3_dir}",
        f"model-path {base}",
    ]))

from src.seer_app import SeerApp
from src.video_processing.environment import AGGREGATION_PATH

app = SeerApp(aggregation_path=AGGREGATION_PATH, debug=False)
app.run()
