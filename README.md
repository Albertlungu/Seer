# Seer

Most tools let you zoom in. Seer lets you zoom all the way in.

The question that started this: _what would it actually feel like to hold an electron?_ Not a diagram. Not a textbook illustration. The real thing — at the scale it exists, with accurate geometry, bonds, orbitals, and structure all the way down.

Seer is a personal experiment to find out. You film a room. The tool reconstructs it as a 3D environment you can walk through. Then you scroll — past the surface of objects, into the molecular structure of the materials they're made of, down to individual atoms, and eventually to fundamental particles. Every scale rendered as accurately as current theory allows.

This is a work in progress. The reconstruction-to-annotation pipeline is working. The visualization side is still being built.

---

## What Works Now

- Video to 3D model via Apple Photogrammetry (macOS only)
- Interactive 3D environment viewer with first-person navigation
- Manual 3D bounding box annotation tool for labeling objects and materials
- LLM-based material composition analysis (Deepseek via Ollama)
- Molecular 3D structure retrieval from PubChem

## Roadmap

- Molecular visualization: rendering atoms and bonds inside the viewer
- Atomic level: Bohr model and quantum mechanical orbital representation
- Subatomic level: protons, neutrons
- Fundamental particles: electrons, quarks, bosons, leptons

---

## Pipeline

```text
Video
  └─ Extract frames          src/video_processing/reconstruction/vid_to_imgs.py
       └─ Apple Photogrammetry → USDZ
            └─ Convert to OBJ + textures   src/video_processing/view_reconstruction.py
                 └─ Annotate objects + materials   src/video_processing/material_tagging/annotator.py
                      └─ LLM: material → molecules   src/video_processing/material_tagging/add_molecular_components.py
                           └─ Fetch 3D structure per molecule   src/render_molecules/mol_details_pubchem.py
                                └─ Interactive viewer   src/video_processing/environment.py  [WIP]
```

---

## Setup

Requires Python 3.12+. The photogrammetry stage requires macOS.

```bash
chmod +x setup.sh
./setup.sh
```

---

## Running Each Stage

### 1. Extract frames from video

Pulls frames from an MP4 at 2 FPS into `data/env_imgs/`.

```bash
python -m src.video_processing.reconstruction.vid_to_imgs
```

### 2. Run Apple Photogrammetry

Calls the native macOS photogrammetry pipeline to produce a USDZ model.

```bash
python -m src.video_processing.reconstruction.run_swift_scan
```

### 3. Convert and inspect the 3D model

Converts USDZ to OBJ, extracts textures, and opens a preview.

```bash
python -m src.video_processing.view_reconstruction
```

### 4. Annotate objects and materials

Interactive tool for drawing 3D bounding boxes directly on the mesh. See controls below.

```bash
source venv/bin/activate
python src/video_processing/material_tagging/annotator.py
```

### 5. Identify molecular composition

Sends annotated materials to a local Deepseek LLM to identify constituent molecules, then looks up their SMILES strings via PubChem.

```bash
python -m src.video_processing.material_tagging.add_molecular_components
```

### 6. Fetch 3D molecular structure

Retrieves atomic coordinates and bond data for each molecule from the PubChem API.

```bash
python -m src.render_molecules.mol_details_pubchem
```

### 7. Open the environment viewer

Loads the OBJ and walks around the reconstructed scene.

```bash
python -m src.video_processing.environment
```

Controls: `WASD` to move, mouse to look, scroll to zoom, `Escape` to toggle mouse lock, `+`/`-` to adjust speed.

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
| `Mouse1` release | on handle | release |
| `Space` | editing | save object name and materials |
| `Tab` | any | discard current box, next color |
| `Ctrl+S` | any | save all annotations |

Annotations are written to `data/vision_json/annotations.json`. Each entry includes the object name, materials, bottom/top corner coordinates, and base normal.

---

## Optional Dependency

`aspose-3d` is only needed for USDZ to OBJ conversion in stage 3. Install it manually if you need that step:

```bash
pip install aspose-3d
```

> [!NOTE]
> On macOS Apple Silicon (`arm64`), a compatible `aspose-3d` wheel may be unavailable. Use an alternative USDZ converter or an `x86_64` Python environment in that case.
