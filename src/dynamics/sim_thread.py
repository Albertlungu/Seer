"""
./src/dynamics/sim_thread.py

Background thread running the MD integration loop.
"""

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from src.dynamics.constants import (
    AMU_TO_KG,
    LANGEVIN_GAMMA,
    MD_TIMESTEP,
    STEPS_PER_BUFFER_WRITE,
)
from src.dynamics.engine import MDEngine
from src.dynamics.integrator import (
    assign_boltzmann_velocities,
    velocity_verlet_step,
)
from src.dynamics.shared_buffer import SharedPositionBuffer
from src.render_molecules.arrangement.geometry import apply_instance_transform
from src.render_molecules.arrangement.scene_state import ObjectState
from src.utils.constants import ELEMENT_MASSES

logger = logging.getLogger(__name__)


@dataclass
class AtomMapping:
    """Bidirectional map between flat simulation array indices and instance atoms."""

    sim_index_to_instance: list[tuple[int, int]]
    """List of (instance_id, local_atom_index) for each row in positions array."""

    instance_to_sim_range: dict[int, tuple[int, int]]
    """Maps instance_id to (start_index, end_index) slice into positions array."""

    total_atoms: int


@dataclass
class SimulationState:
    """Full mutable state of the MD simulation."""

    positions: np.ndarray  # (N, 3) metres
    velocities: np.ndarray  # (N, 3) m/s
    forces: np.ndarray  # (N, 3) Newtons
    masses: np.ndarray  # (N,) kg
    atomic_numbers: np.ndarray  # (N,) ints
    temperature: float  # Kelvin
    timestep: float  # seconds
    step_count: int = 0
    running: bool = False
    error: str | None = None


def build_atom_mapping(
    object_state: ObjectState,
    active_instance_ids: list[int],
) -> AtomMapping:
    """
    Build the flat-array index mapping for a set of active instances.

    Args:
        object_state: Scene state with templates and instances.
        active_instance_ids: Which instances to include.

    Returns:
        AtomMapping covering only the active instances.
    """
    index_list: list[tuple[int, int]] = []
    range_map: dict[int, tuple[int, int]] = {}
    offset = 0

    for iid in active_instance_ids:
        inst = object_state.instances[iid]
        tmpl = object_state.templates[inst.template_id]
        n_atoms = len(tmpl.aids)
        range_map[iid] = (offset, offset + n_atoms)
        for local_idx in range(n_atoms):
            index_list.append((iid, local_idx))
        offset += n_atoms

    return AtomMapping(
        sim_index_to_instance=index_list,
        instance_to_sim_range=range_map,
        total_atoms=offset,
    )


def flatten_positions(
    object_state: ObjectState,
    atom_mapping: AtomMapping,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract world-space atom positions, masses, and atomic numbers into flat arrays.

    Args:
        object_state: Scene state.
        atom_mapping: Index mapping.

    Returns:
        Tuple of (positions (N,3), masses (N,), atomic_numbers (N,)).
    """
    n = atom_mapping.total_atoms
    positions = np.empty((n, 3), dtype=np.float64)
    masses = np.empty(n, dtype=np.float64)
    atomic_numbers = np.empty(n, dtype=np.int64)

    for iid, (start, end) in atom_mapping.instance_to_sim_range.items():
        inst = object_state.instances[iid]
        tmpl = object_state.templates[inst.template_id]
        world_xyz = apply_instance_transform(template=tmpl, instance=inst)
        coords = np.column_stack(world_xyz)  # (n_atoms, 3)
        positions[start:end] = coords

        for local_idx, global_idx in enumerate(range(start, end)):
            element = int(tmpl.elements[local_idx])
            atomic_numbers[global_idx] = element
            mass_amu = ELEMENT_MASSES.get(element, 12.0)
            masses[global_idx] = mass_amu * AMU_TO_KG

    return positions, masses, atomic_numbers


class SimulationThread:
    """
    Runs the MD loop on a background daemon thread.

    Usage:
        sim = SimulationThread(engine, object_state, active_ids)
        sim.start()
        sim.pause()
        sim.resume()
        sim.stop()
        sim.set_temperature(500.0)
    """

    def __init__(
        self,
        engine: MDEngine,
        object_state: ObjectState,
        active_instance_ids: list[int],
        temperature: float = 298.15,
    ) -> None:
        self.engine: MDEngine = engine
        self.object_state: ObjectState = object_state

        self._mapping: AtomMapping = build_atom_mapping(
            object_state, active_instance_ids
        )
        positions, masses, atomic_numbers = flatten_positions(
            object_state, self._mapping
        )

        rng = np.random.default_rng(42)
        velocities = assign_boltzmann_velocities(masses, temperature, rng)
        forces, _ = engine.evaluate_forces(positions, atomic_numbers)

        self.state: SimulationState = SimulationState(
            positions=positions,
            velocities=velocities,
            forces=forces,
            masses=masses,
            atomic_numbers=atomic_numbers,
            temperature=temperature,
            timestep=MD_TIMESTEP,
        )

        self.buffer: SharedPositionBuffer = SharedPositionBuffer(
            self._mapping.total_atoms
        )
        self.buffer.write(positions)

        self._rng: np.random.Generator = rng
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event = threading.Event()
        self._paused_event: threading.Event = threading.Event()
        self._paused_event.set()  # Start paused; user must click toggle

    @property
    def mapping(self) -> AtomMapping:
        return self._mapping

    def start(self) -> None:
        """Spawn the background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.state.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the thread to exit and wait for it."""
        self._stop_event.set()
        self._paused_event.clear()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.state.running = False

    def pause(self) -> None:
        """Pause the simulation loop. State is preserved."""
        self._paused_event.set()

    def resume(self) -> None:
        """Resume the simulation loop from current state."""
        self._paused_event.clear()

    def is_running(self) -> bool:
        return (
            self._thread is not None
            and self._thread.is_alive()
            and not self._paused_event.is_set()
        )

    def set_temperature(self, temperature: float) -> None:
        """Update the thermostat target. Takes effect on the next step."""
        self.state.temperature = max(0.0, temperature)

    def _run(self) -> None:
        """Main loop executed on the background thread."""
        while not self._stop_event.is_set():
            if self._paused_event.is_set():
                time.sleep(0.001)
                continue

            try:
                for _ in range(STEPS_PER_BUFFER_WRITE):
                    (
                        self.state.positions,
                        self.state.velocities,
                        self.state.forces,
                    ) = velocity_verlet_step(
                        positions=self.state.positions,
                        velocities=self.state.velocities,
                        forces=self.state.forces,
                        masses=self.state.masses,
                        dt=self.state.timestep,
                        temperature=self.state.temperature,
                        gamma=LANGEVIN_GAMMA,
                        engine=self.engine,
                        atomic_numbers=self.state.atomic_numbers,
                        rng=self._rng,
                    )
                    self.state.step_count += 1

                self.buffer.write(self.state.positions)

            except Exception as exc:
                logger.error("MD thread error: %s", exc, exc_info=True)
                self.state.error = str(exc)
                self.state.running = False
                self._paused_event.set()
                break
