# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Seer.

Build:
    pyinstaller seer.spec

Produces:
    dist/Seer.app     (macOS)
    dist/Seer/        (Windows / Linux — zip or tar for distribution)
"""

import glob
import os
import panda3d
from PyInstaller.utils.hooks import collect_dynamic_libs

_pd3_dir = os.path.dirname(panda3d.__file__)
_pd3_models = os.path.join(_pd3_dir, "models")
_pd3_etc = os.path.join(_pd3_dir, "etc")

# Collect all Panda3D runtime plugins (.dylib / .dll / .so)
_pd3_libs = collect_dynamic_libs("panda3d")

block_cipher = None

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=_pd3_libs,
    datas=[
        # Room geometry and molecular data
        ("data/reconstructions/bam/albert_room.bam", "data/reconstructions/bam"),
        # Textures referenced by the bam as ../obj/*.png
        *[(f, "data/reconstructions/obj") for f in glob.glob("data/reconstructions/obj/*.png")],
        ("data/vision_json/final_aggregated.json",   "data/vision_json"),
        # Panda3D built-in models (sphere, box, etc.)
        (_pd3_models, "panda3d/models"),
        # Panda3D config (Config.prc sets load-display pandagl)
        (_pd3_etc,    "panda3d/etc"),
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
