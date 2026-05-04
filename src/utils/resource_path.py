"""
./src/utils/resource_path.py

Resolves data-file paths in both development (running from source) and
production (PyInstaller bundle) modes.
"""

import os
import sys


def resource_path(relative: str) -> str:
    """
    Return the absolute path to a bundled data file.

    In a PyInstaller bundle, files land in sys._MEIPASS.
    In development, paths resolve relative to the project root.

    Args:
        relative: Path relative to the project root (e.g. "data/vision_json/foo.json").

    Returns:
        Absolute path usable by open() or Panda3D's loadModel().
    """
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS  # type: ignore[attr-defined]
    else:
        # src/utils/ -> src/ -> project root
        base = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.normpath(os.path.join(base, relative)).replace("\\", "/")
