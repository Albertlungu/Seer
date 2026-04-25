"""
Seer application entry point.

Run from source:  python main.py
Bundled:          Seer.exe / Seer.app / Seer (Linux)
"""

import os
import sys

# When bundled with PyInstaller, tell Panda3D where its model files landed.
if getattr(sys, "frozen", False):
    from panda3d.core import loadPrcFileData

    model_dir = os.path.join(sys._MEIPASS, "panda3d", "models")  # type: ignore[attr-defined]
    loadPrcFileData("", f"model-path {model_dir}")

from src.seer_app import SeerApp
from src.video_processing.environment import AGGREGATION_PATH

app = SeerApp(aggregation_path=AGGREGATION_PATH, debug=False)
app.run()
