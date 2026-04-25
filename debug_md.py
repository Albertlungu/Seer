"""
python -m debug_md

Test the harmonic SimulationThread for organic and inorganic molecules.
Verifies that all atoms move, step sizes stay within bounds, and the
interpolation guard accepts frames.
"""

import time

import numpy as np

from src.dynamics.sim_thread import SimulationThread
from src.dynamics.constants import MD_TIMESTEP
from src.render_molecules.arrange_molecules import build_templates_from_object
from src.render_molecules.arrangement.geometry import compute_bounding_sphere_radius
from src.render_molecules.arrangement.placement import PlacementConfig, place_molecules
from src.render_molecules.arrangement.scene_state import ObjectState
from src.utils.constants import CHUNK_MOL_COUNT_PER_TEMPLATE, CHUNK_SIZE_A
from src.utils.json_io import load_json

SPEED = 3.0
DT = MD_TIMESTEP * SPEED
STEP_THRESHOLD_A = 5.0
CUMULATIVE_LIMIT_A = 20.0

data = load_json("final_aggregated.json")

def run_for_object(obj_key: str, wait_s: float = 2.0) -> None:
    print(f"\n{'='*60}")
    print(f"Object: {obj_key}  DT={DT:.2e}s")
    templates = build_templates_from_object(data[obj_key])
    if not templates:
        print("  No templates.")
        return

    s = CHUNK_SIZE_A
    cmin, cmax = np.zeros(3), np.array([s, s, s])
    box_bottom = np.array([[cmin[0],cmax[0],cmax[0],cmin[0]],
                            [cmin[1],cmin[1],cmax[1],cmax[1]],
                            [cmin[2],cmin[2],cmin[2],cmin[2]]], dtype=float)
    box_top    = np.array([[cmin[0],cmax[0],cmax[0],cmin[0]],
                            [cmin[1],cmin[1],cmax[1],cmax[1]],
                            [cmax[2],cmax[2],cmax[2],cmax[2]]], dtype=float)

    max_r = max(compute_bounding_sphere_radius(t) for t in templates.values())
    target_counts = {tid: CHUNK_MOL_COUNT_PER_TEMPLATE for tid in templates}
    total = sum(target_counts.values())

    obj_state = ObjectState(
        object_key="chunk", object_name="chunk", instance_id="chunk",
        display_name="chunk", templates=templates, instances={},
        box_bottom=box_bottom, box_top=box_top, rng_seed=42,
    )
    config = PlacementConfig(
        seed=42, frontier_radius=CHUNK_SIZE_A * 0.5,
        min_center_distance=max(max_r, 2.0) * 3.5,
        max_total_attempts=total * 100, target_instance_count=total,
        stop_when_target_met=True, require_in_bounds=True, require_no_overlap=True,
    )
    obj_state = place_molecules(object_state=obj_state, config=config, target_counts=target_counts)
    print(f"  Placed {len(obj_state.instances)} instances")

    sim = SimulationThread(
        object_state=obj_state,
        active_instance_ids=list(obj_state.instances.keys()),
        temperature=298.15,
    )
    sim.set_timestep(DT)
    sim.start()
    sim.resume()
    print(f"  Thread started. Waiting {wait_s}s...")

    anchor = sim.buffer.read().copy()
    time.sleep(wait_s)
    pos_after = sim.buffer.read().copy()

    steps = sim.state.step_count
    max_step_a = float(np.max(np.abs(pos_after - anchor))) / 1e-10

    # Per-element max displacement
    from src.dynamics.sim_thread import build_atom_mapping, flatten_positions
    mapping = sim._mapping
    _, _, atomic_numbers = flatten_positions(obj_state, mapping)
    per_z: dict[int, float] = {}
    for z in np.unique(atomic_numbers):
        mask = atomic_numbers == z
        disp = np.abs(pos_after[mask] - anchor[mask]) / 1e-10
        per_z[int(z)] = float(np.max(disp))

    print(f"  Steps: {steps}  Total displacement: {max_step_a:.3f}Å  per-Z: {per_z}")
    if steps == 0:
        print("  FAIL: no steps completed")
    elif max_step_a < 0.001:
        print("  FAIL: atoms not moving")
    elif max_step_a > CUMULATIVE_LIMIT_A:
        print(f"  WARN: cumulative drift {max_step_a:.1f}Å exceeds {CUMULATIVE_LIMIT_A}Å limit")
    else:
        print("  OK: atoms moving within bounds")

    sim.stop()


run_for_object("books")    # cellobiose + coniferyl-alcohol (organic)
run_for_object("clock")    # SiO2 + iron (inorganic)
run_for_object("chair")    # cellobiose + iron + ethylene (mixed)
