# Contributing to Seer

## Getting Started

Requires Python 3.12 and MacOS for the photogrammetry pipeline. The viewer and molecular simulation themselves run on all platforms.

```bash
git clone https://github.com/Albertlungu/Seer.git
cd Seer
chmod +x setup.sh
./setup.sh
source venv/bin/activate
python main.py
```

## Project structure
```
src/
  seer_app.py                    — main app entrypoint, orchestrates everything
  dynamics/                      — molecular simulation
    sim_thread.py                — background MD thread (harmonic bond springs)
    engine.py                    — MACE force evaluation (future use)
    integrator.py                — Langevin integrator (BAOAB)
    shared_buffer.py             — double-buffered position transfer
    constants.py                 — physical constants
  render_molecules/
    arrange_molecules.py         — builds molecule templates from JSON
    arrangement/
      placement.py               — frontier-based molecule placement
      renderer.py                — Panda3D scene graph rendering
      scene_state.py             — ObjectState, MoleculeTemplate, MoleculeInstance
      geometry.py                — geometric helpers
  video_processing/
    environment.py               — 3D room environment, RoomState, movement
    material_tagging/
      annotator.py               — interactive 3D bounding box annotation tool
      add_molecular_components.py — LLM-based material composition analysis
  utils/
    constants.py                 — app-wide constants (chunk size, scale, etc.)
    resource_path.py             — resolves data paths in dev and bundled modes
    json_io.py                   — JSON load/save helpers
  zoom/
    raycast_picker.py            — screen-center raycast for object selection
    zoom_controller.py           — FOV-based zoom state machine
data/
  reconstructions/               — room geometry (.bam + textures)
  vision_json/                   — aggregated molecular data (final_aggregated.json)
```

## Running the Debug Scripts

If something goes wrong in the dynamics portion of it, you may want to first run the debug script to figure out exactly what is going wrong and where:

```bash
python -m debug_md
```

This tests template loading, molecule placement, the simulation thread lifecycle, and reports per-element displacements. If all three objects show "OK: Atoms moving within bounds", you're good.

## Key Constants

All tunable parameters are in `src/utils/constants/py`.

**Chunk streaming:**

- `CHUNK_SIZE_A`: side length of one chunk in Angstroms (default 100Å = 1m visual)
- `LOAD_RADIUS_CHUNKS`: How many chunks to load in each direction from the camera
- `CHUNK_MOL_COUNT_PER_TEMPLATE`: molecules per template per chunk (density)
- `WORLD_CHUNKS`: world size before it loops back around
- `MOL_VIEW_SCALE`: Angstroms-to-metres conversion for the mol_root scale (0.01)

**Simulation:**

- `MD_TIMESTEP`: base integration timestep. The speed slider multiplies this.
- `LANGEVIN_GAMMA`: Langevin thermostat collision frequency
- `SimulationThread.HARMONIC_DT`: internal fixed timestep for the harmonic integrator (1fs, DO NOT MODIFY)
- `SimulationThread.K_BOND`: bond spring constant (N/m)
- `SimulationThread.K_ANCHOR`: weak per-atom anchor spring (N/m)

## Physics Notes

The current simulation uses harmonic bond springs rather than MACE forces. Each bonded atom has a spring at 200 N/m connecting it, which gives ~0.14 Angstrom bond length fluctuations at 298K. A weak per-atom anchor (0.5 N/m) prevents unlimited drift. the internal timestep is fixed at 1fs regardless of the speed slider. That controls the steps per buffer write.

MACE (the ML force field) is in the codebase (`src/dynamics/engine.py`) but is unused in the simulation loop. At `MD_TIMESTEP = 5e-14 s`, C-H bond vibrations alias and the integration blows up. The path back to real forces is reducing the timestep to 1-2 fs, then re-enabling. It is disabled for now, since it was too unstable.

## Code Style

Follow whatever is already in the file you're editing. A few project-specific things:

- No emojis anywhere
- All function signatures and class attributes have type annotations
- Comments only when the why is non-obvious. Never restate what the code does
- No docstring padding ("This function does X" when the name already says X)

## Building a Release

Test the bundle locally before pushing a tag:

```bash
pip install pyinstaller pyinstaller-hooks-contrib
pyinstaller seer.spec --noconfirm
dist/Seer.app/Contents/MacOS/Seer
```

If the app opens and the room renders correctly, push the tag:

```bash
git tag v1.x.x && git push origin v1.x.x
```

GitHub Actions will build macOS (.dmg), Windows (.zip), and Linux (.tar.gz) and publish them as a release.

## Data Files

The following files are not tracked by git and need to be generated locally before running the full pipeline:

- `data/reconstructions/obj/` — your room scan (OBJ + textures)
- `data/reconstructions/bam/` — Panda3D native format converted from OBJ
- `data/vision_json/annotations.json` — bounding box annotations
- `data/vision_json/aggregated.json` — LLM material composition output
- `data/vision_json/final_aggregated.json` — final merged output with PubChem data

The `final_aggregated.json` for the demo room is committed and bundled with releases. To use your own room, run the full pipeline from step 1 in the README.
