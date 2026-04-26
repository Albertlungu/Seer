"""
./src/dynamics/sim_thread.py

Background thread running the MD integration loop.
"""

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from src.utils.constants import (
    AMU_TO_KG,
    HARMONIC_DT,
    K_ANCHOR,
    K_BOND,
    LANGEVIN_GAMMA,
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
        coords = np.column_stack(world_xyz)  # (n_atoms, 3) in Angstroms
        positions[start:end] = coords * 1e-10  # Å -> metres

        for local_idx, global_idx in enumerate(range(start, end)):
            element = int(tmpl.elements[local_idx])
            atomic_numbers[global_idx] = element
            mass_amu = ELEMENT_MASSES.get(element, 12.0)
            # Use carbon mass for H in the harmonic integrator: real H mass (1 amu)
            # causes float64 overflow in the velocity update at dt > ~20 fs.
            if element == 1:
                mass_amu = 12.0
            masses[global_idx] = mass_amu * AMU_TO_KG

    return positions, masses, atomic_numbers


class SimulationThread:
    """
    Runs a bond-spring harmonic MD loop on a background daemon thread.

    Forces:
      Bond springs  F_bond = K_BOND * (r_ij - r_ij_eq)     — preserves molecular shape
      Atom anchor   F_anchor = -K_ANCHOR * (x - x_eq)       — prevents unlimited drift
      Langevin thermostat                                     — controls temperature

    Internal timestep is fixed at 1 fs for stability regardless of the speed
    slider. set_timestep() maps the slider value to steps_per_write, controlling
    how many fs of simulation time are advanced each buffer write (visual speed).
    """

    def __init__(
        self,
        object_state: ObjectState,
        active_instance_ids: list[int],
        temperature: float = 298.15,
        engine: MDEngine | None = None,
    ) -> None:
        self.engine: MDEngine | None = engine
        self.object_state: ObjectState = object_state

        self._mapping: AtomMapping = build_atom_mapping(
            object_state, active_instance_ids
        )
        positions, masses, atomic_numbers = flatten_positions(
            object_state, self._mapping
        )

        self._equilibrium: np.ndarray = positions.copy()

        # Precompute bond spring data from molecular topology.
        # For each bond: (flat_i, flat_j) index pair and equilibrium bond vector.
        f1_list: list[int] = []
        f2_list: list[int] = []
        for iid, (start, _end) in self._mapping.instance_to_sim_range.items():
            inst = object_state.instances[iid]
            tmpl = object_state.templates[inst.template_id]
            aid_to_local = {int(aid): i for i, aid in enumerate(tmpl.aids)}
            for a1, a2 in zip(tmpl.bonds_aid1, tmpl.bonds_aid2):
                l1 = aid_to_local.get(int(a1))
                l2 = aid_to_local.get(int(a2))
                if l1 is not None and l2 is not None:
                    f1_list.append(start + l1)
                    f2_list.append(start + l2)

        self._bond_f1: np.ndarray = np.array(f1_list, dtype=np.intp)
        self._bond_f2: np.ndarray = np.array(f2_list, dtype=np.intp)
        # Equilibrium bond vectors: r_j_eq - r_i_eq for each bond
        if len(f1_list):
            self._bond_eq: np.ndarray = (
                positions[self._bond_f2] - positions[self._bond_f1]
            )
        else:
            self._bond_eq = np.zeros((0, 3))

        # Steps per buffer write; controlled by set_timestep via speed slider
        self._steps_per_write: int = 50

        rng = np.random.default_rng(42)
        velocities = assign_boltzmann_velocities(masses, temperature, rng)

        self.state: SimulationState = SimulationState(
            positions=positions,
            velocities=velocities,
            forces=np.zeros_like(positions),
            masses=masses,
            atomic_numbers=atomic_numbers,
            temperature=temperature,
            timestep=HARMONIC_DT,
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
        """Signal the thread to exit without blocking the caller.

        Setting the stop event lets the daemon thread exit asynchronously. We avoid
        joining here to prevent the main/UI thread from blocking when many
        simulation threads are stopped at once.
        """
        self._stop_event.set()
        self._paused_event.clear()
        # Mark as not running immediately so callers observe the stopped state.
        self.state.running = False
        # Do not join the thread here; it is a daemon and will exit on its own.
        # If the thread has already finished, clear the reference.
        if self._thread is not None and not self._thread.is_alive():
            self._thread = None

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

    def set_timestep(self, dt: float) -> None:
        """Map speed-slider value to steps_per_write for the harmonic integrator."""
        self._steps_per_write = max(1, int(dt / HARMONIC_DT))

    def _compute_forces(self, positions: np.ndarray) -> np.ndarray:
        """
        Bond springs + weak per-atom anchor.

        Bond springs keep bond lengths near equilibrium (~0.14 Å σ at 298 K),
        preserving the molecule's shape. The anchor (K_ANCHOR << K_BOND) prevents
        the whole molecule from drifting arbitrarily far.
        """
        forces = -K_ANCHOR * (positions - self._equilibrium)

        if len(self._bond_f1):
            delta = (
                positions[self._bond_f2] - positions[self._bond_f1] - self._bond_eq
            )
            bond_f = K_BOND * delta
            np.add.at(forces, self._bond_f1, bond_f)
            np.add.at(forces, self._bond_f2, -bond_f)

        return forces

    def _run(self) -> None:
        """Main loop: run _steps_per_write BAOAB steps then write buffer once."""
        from src.dynamics.integrator import langevin_half_kick

        while not self._stop_event.is_set():
            if self._paused_event.is_set():
                time.sleep(0.001)
                continue

            try:
                dt = HARMONIC_DT
                pos = self.state.positions
                vel = self.state.velocities
                masses = self.state.masses
                T = self.state.temperature

                for _ in range(self._steps_per_write):
                    f = self._compute_forces(pos)
                    vel = langevin_half_kick(vel, f, masses, dt, T, LANGEVIN_GAMMA, self._rng)
                    pos = pos + dt * vel
                    f = self._compute_forces(pos)
                    vel = langevin_half_kick(vel, f, masses, dt, T, LANGEVIN_GAMMA, self._rng)

                self.state.positions = pos
                self.state.velocities = vel
                self.state.forces = self._compute_forces(pos)
                self.state.step_count += self._steps_per_write
                self.buffer.write(pos)

            except Exception as exc:
                logger.error("MD thread error: %s", exc, exc_info=True)
                self.state.error = str(exc)
                self.state.running = False
                self._paused_event.set()
                break
