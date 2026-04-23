"""
./src/dynamics/constants.py

Physical constants and crystal structure data for the MD engine.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------

MD_TIMESTEP: float = 5e-14
"""Integration timestep in seconds. 0.5 femtoseconds."""

BOLTZMANN_CONSTANT: float = 1.380649e-23
"""Boltzmann constant in J/K. Exact by 2019 SI redefinition."""

LANGEVIN_GAMMA: float = 1e13
"""Langevin thermostat collision frequency in s^-1. Gives ~100 fs relaxation time."""

DEFAULT_TEMPERATURE: float = 298.15
"""Room temperature in Kelvin."""

MIN_TEMPERATURE: float = 100.0
"""Minimum slider value in Kelvin."""

MAX_TEMPERATURE: float = 1000.0
"""Maximum slider value in Kelvin."""

EV_TO_JOULE: float = 1.602176634e-19
"""Conversion factor from electronvolts to joules. Exact by SI definition."""

ANGSTROM_TO_METRE: float = 1e-10
"""Conversion factor from Angstroms to metres."""

STEPS_PER_BUFFER_WRITE: int = 100
"""Number of MD steps between each shared buffer write. Tune for framerate."""

AMU_TO_KG: float = 1.66053906660e-27
"""Conversion factor from atomic mass units (g/mol) to kilograms per atom."""

# ---------------------------------------------------------------------------
# Crystal structures: atomic_number -> (type, lattice_param_m, basis_fractional)
# ---------------------------------------------------------------------------

CRYSTAL_STRUCTURES: dict[int, tuple[str, float, list[list[float]]]] = {
    13: (
        "fcc",
        4.0495e-10,
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    ),
    14: (
        "diamond",
        5.4310e-10,
        [
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ],
    ),
    20: (
        "fcc",
        5.5884e-10,
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    ),
    26: (
        "bcc",
        2.8665e-10,
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    ),
    29: (
        "fcc",
        3.6149e-10,
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    ),
}

DEFAULT_CRYSTAL_TYPE: str = "fcc"
"""Fallback crystal type for unlisted metallic elements."""
