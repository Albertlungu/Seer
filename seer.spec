# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Seer.

Build:
    pyinstaller seer.spec

Produces:
    dist/Seer.app     (macOS)
    dist/Seer/        (Windows / Linux — zip or tar for distribution)
"""

import os
import panda3d

_pd3_models = os.path.join(os.path.dirname(panda3d.__file__), "models")

block_cipher = None

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=[
        # Room geometry and molecular data
        ("data/reconstructions/obj/albert_room.obj", "data/reconstructions/obj"),
        ("data/vision_json/final_aggregated.json",   "data/vision_json"),
        # Panda3D built-in models (sphere, box, etc.)
        (_pd3_models, "panda3d/models"),
    ],
    hiddenimports=[
        "panda3d.core",
        "direct.showbase.ShowBase",
        "direct.showbase.ShowBaseGlobal",
        "direct.task",
        "direct.task.Task",
        "direct.gui.DirectGui",
        "direct.gui.DirectCheckButton",
        "direct.gui.DirectSlider",
        "direct.gui.DirectLabel",
        "src.dynamics",
        "src.dynamics.sim_thread",
        "src.dynamics.integrator",
        "src.dynamics.shared_buffer",
        "src.dynamics.constants",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # ML stack — not needed for harmonic dynamics build
        "torch", "torchvision", "torchaudio",
        "mace", "ase", "e3nn",
        "scipy",
        "matplotlib", "PIL", "cv2",
        "open3d",
        "IPython", "jupyter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Seer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,        # no terminal window
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Seer",
)

# macOS .app bundle (ignored on Windows / Linux)
app = BUNDLE(
    coll,
    name="Seer.app",
    bundle_identifier="com.seer.molecular",
    info_plist={
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
        "CFBundleShortVersionString": "1.0.0",
    },
)
