"""
./src/dynamics/integrator.py

Velocity Verlet integrator with Langevin thermostat (BAOAB splitting).
All quantities in SI units (metres, seconds, kilograms, joules).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.dynamics.constants import BOLTZMANN_CONSTANT

if TYPE_CHECKING:
    from src.dynamics.engine import MDEngine


def assign_boltzmann_velocities(
    masses: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample velocities from the Maxwell-Boltzmann distribution.

    Args:
        masses: Array of shape (N,) in kilograms.
        temperature: Target temperature in Kelvin.
        rng: Numpy random generator.

    Returns:
        Velocities of shape (N, 3) in m/s with zero total momentum.
    """
    if temperature <= 0:
        return np.zeros((len(masses), 3), dtype=np.float32)

    sigma = np.sqrt(BOLTZMANN_CONSTANT * temperature / masses)[:, None]
    v = rng.standard_normal((len(masses), 3)) * sigma

    # Zero total momentum to prevent centre-of-mass drift
    total_momentum = np.sum(masses[:, None] * v, axis=0)
    v -= total_momentum / np.sum(masses)

    return v


def langevin_half_kick(
    velocities: np.ndarray,
    forces: np.ndarray,
    masses: np.ndarray,
    dt: float,
    temperature: float,
    gamma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    One half-step of the BAOAB Langevin integrator (the O step plus half B step).

    Args:
        velocities: Current velocities, shape (N, 3) in m/s.
        forces: Current forces, shape (N, 3) in Newtons.
        masses: Atom masses, shape (N,) in kg.
        dt: Full timestep in seconds.
        temperature: Thermostat target in Kelvin.
        gamma: Langevin collision frequency in s^-1.
        rng: Numpy random generator.

    Returns:
        Updated velocities, shape (N, 3).
    """
    inv_mass = (1.0 / masses)[:, None]

    # B step: half-kick from forces
    velocities = velocities + 0.5 * dt * forces * inv_mass

    # O step: Langevin friction and noise
    c1 = np.exp(-gamma * dt)
    c2_sq = (1.0 - c1 * c1) * BOLTZMANN_CONSTANT * temperature
    c2 = np.sqrt(np.maximum(c2_sq, 0.0)) * np.sqrt(inv_mass)
    noise = rng.standard_normal(velocities.shape)
    velocities = c1 * velocities + c2 * noise

    return velocities


def velocity_verlet_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    masses: np.ndarray,
    dt: float,
    temperature: float,
    gamma: float,
    engine: MDEngine,
    atomic_numbers: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One full BAOAB Velocity Verlet step with Langevin thermostat.

    Sequence: half-kick -> drift -> force eval -> half-kick.

    Args:
        positions: Atom positions, shape (N, 3) in metres.
        velocities: Atom velocities, shape (N, 3) in m/s.
        forces: Current forces, shape (N, 3) in Newtons.
        masses: Atom masses, shape (N,) in kg.
        dt: Timestep in seconds.
        temperature: Thermostat target in Kelvin.
        gamma: Langevin collision frequency in s^-1.
        engine: MDEngine instance for force evaluation.
        atomic_numbers: Element numbers, shape (N,).
        rng: Numpy random generator.

    Returns:
        Tuple of (new_positions, new_velocities, new_forces).
    """
    velocities = langevin_half_kick(
        velocities, forces, masses, dt, temperature, gamma, rng
    )

    positions = positions + dt * velocities

    new_forces, _energy = engine.evaluate_forces(positions, atomic_numbers)

    velocities = langevin_half_kick(
        velocities, new_forces, masses, dt, temperature, gamma, rng
    )

    return positions, velocities, new_forces
