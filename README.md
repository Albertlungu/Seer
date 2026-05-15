# Seer

Most tools let you zoom in. Seer lets you zoom all the way in.

You walk around a real 3D room scan, zoom into any object, and descend into its actual molecular structure — the real chemical composition of the material, sourced from PubChem. Atoms vibrate with thermal dynamics at adjustable temperature. Walk in any direction and the molecular world keeps generating around you. The world loops so you never hit a wall.

Available for macOS, Windows, and Linux. [Download the latest release.](../../releases/latest)

---

## What Works Now

- First-person 3D environment from a real room scan
- Zoom from room scale to molecular scale by scrolling
- Infinite procedurally-streamed molecular world (chunk-based, deterministic)
- Real material composition from PubChem for each object
- Electron density bond clouds (sigma and pi orbital structure)
- Molecular dynamics: atoms vibrate with Langevin thermostat at adjustable temperature
- Stick bonds during dynamics, clouds when paused
- Atom scale slider, temperature slider, speed slider
- Toroidal world boundary (walking off one edge brings you back from the other)
- Cross-platform packaged releases (macOS .dmg, Windows .zip, Linux .tar.gz)

## Roadmap

- MACE force field at stable timestep for physically accurate inter-atomic forces
- Atomic level: quantum mechanical orbital representation
- Subatomic level: protons, neutrons, fundamental particles

---

## Controls

| Input | Action |
| --- | --- |
| `WASD` | Move |
| Mouse | Look |
| Scroll down | Zoom in (enters molecular mode when close enough) |
| Scroll up | Zoom out (exits molecular mode) |
| `Escape` | Toggle mouse lock |
| `+` / `-` | Adjust movement speed |

When in molecular mode, a panel appears on the right:

| Control | Effect |
| --- | --- |
| **Time** toggle | Start / pause molecular dynamics |
| **Temp** slider | Thermostat target (100–1000 K) |
| **Speed** slider | Simulation speed (steps per frame) |
| **Electron Clouds** toggle | Show/hide bond clouds |
| **Atom Scale** slider | Slide between covalent and van der Waals radii |

---

## Using Your Own Room

### 1. Film your room

Record three slow passes around the perimeter of the room on your phone, one at each height: low (near floor), mid (waist), and high (near ceiling). Walk slowly and keep the camera pointed at the walls and objects. Overlap each pass generously. MP4 works fine.

### 2. Extract frames

```bash
python -m src.video_processing.reconstruction.vid_to_imgs
```

Pulls frames at 2 FPS into `data/env_imgs/`.

### 3. Run Apple Photogrammetry (macOS only)

```bash
python -m src.video_processing.reconstruction.run_swift_scan
```

Uses the native macOS photogrammetry pipeline to produce a USDZ model in `data/reconstructions/`.

### 4. Convert to OBJ

```bash
python -m src.video_processing.view_reconstruction
```

Converts USDZ to OBJ + texture files in `data/reconstructions/obj/`.

### 5. Convert OBJ to BAM

The viewer loads Panda3D's native `.bam` format to avoid a runtime assimp issue in the packaged app.

```bash
python -c "
from panda3d.core import loadPrcFileData
loadPrcFileData('', 'window-type offscreen')
from direct.showbase.ShowBase import ShowBase
base = ShowBase()
node = base.loader.loadModel('data/reconstructions/obj/albert_room.obj')
node.writeBamFile('data/reconstructions/bam/albert_room.bam')
base.destroy()
print('Done')
"
```

Then update the path in `src/video_processing/environment.py` and `seer.spec` to point to your new `.bam` file.

### 6. Annotate objects and materials

Draw 3D bounding boxes on the mesh and label each object's material.

```bash
source venv/bin/activate
python src/video_processing/material_tagging/annotator.py
```

See annotator controls below.

### 7. Identify molecular composition

Sends material names to a local Deepseek LLM via Ollama to identify constituent molecules, then fetches their SMILES strings from PubChem.

```bash
python -m src.video_processing.material_tagging.add_molecular_components
```

### 8. Fetch 3D molecular structure

Retrieves atomic coordinates and bond data per molecule from the PubChem REST API.

```bash
python -m src.render_molecules.processing.mol_details_pubchem
```

### 9. Run the viewer

```bash
python main.py
```

---

## Setup

Requires Python 3.12.

```bash
chmod +x setup.sh
./setup.sh
```

---

## Building a Release

```bash
pip install pyinstaller pyinstaller-hooks-contrib
pyinstaller seer.spec
```

The result is in `dist/Seer.app` (macOS), `dist/Seer/` (Windows/Linux). To create a GitHub release for all three platforms, push a version tag:

```bash
git tag v1.0.0 && git push origin v1.0.0
```

GitHub Actions will build and attach the installers automatically.

---

## Annotator Controls

| Input | State | Action |
| --- | --- | --- |
| `E` | any | toggle annotation mode |
| `Mouse1` | idle | place anchor on mesh surface |
| mouse move | drawing | preview base rectangle |
| `Mouse1` | drawing | lock base rectangle |
| mouse move | height | preview extrusion height |
| `Mouse1` | height | lock height, spawn corner handles |
| `Mouse1` + drag | on handle | resize box |
| `Space` | editing | save object name and materials |
| `Tab` | any | discard current box |
| `Ctrl+S` | any | save all annotations |

Annotations are saved to `data/vision_json/annotations.json`.

---

## Pipeline

```text
Video
  └─ Extract frames          src/video_processing/reconstruction/vid_to_imgs.py
       └─ Apple Photogrammetry → USDZ
            └─ Convert to OBJ + textures   src/video_processing/view_reconstruction.py
                 └─ Convert OBJ to BAM     (one-time python command above)
                      └─ Annotate objects + materials   src/video_processing/material_tagging/annotator.py
                           └─ LLM: material → molecules   src/video_processing/material_tagging/add_molecular_components.py
                                └─ Fetch 3D structure per molecule   src/render_molecules/processing/mol_details_pubchem.py
                                     └─ Interactive viewer   main.py
```

---

## Optional Dependency

`aspose-3d` is only needed for USDZ to OBJ conversion. Install it manually if needed:

```bash
pip install aspose-3d
```

> [!NOTE]
> On macOS Apple Silicon, a compatible `aspose-3d` wheel may be unavailable. Use an alternative USDZ converter or an `x86_64` Python environment in that case.

**He is risen.**
