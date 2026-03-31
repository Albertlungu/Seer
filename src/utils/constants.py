"""
./src/utils/constants.py

Stores all constants used in the whole project.
"""

FINAL_AGGREGATED = "final_aggregated.json"

ELEMENT_COLORS: dict[int, tuple[float, float, float]] = {
    1: (1.0, 1.0, 1.0),  # Hydrogen
    6: (0.3, 0.3, 0.3),  # Carbon
    7: (0.2, 0.4, 1.0),  # Nitrogen
    8: (1.0, 0.2, 0.2),  # Oxygen
    14: (0.7, 0.5, 0.1),  # Silicon
    16: (1.0, 0.9, 0.0),  # Sulfur
    20: (0.5, 0.5, 0.5),  # Calcium
    26: (0.6, 0.3, 0.1),  # Iron
}

ELEMENT_RADII: dict[int, float] = {
    1: 0.05,
    6: 0.08,
    7: 0.08,
    8: 0.08,
    14: 0.10,
    16: 0.10,
    20: 0.12,
    26: 0.12,
}

ELEMENT_MASSES: dict[int, float] = {
    1: 1.008,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    14: 28.085,
    16: 32.060,
    20: 40.078,
    26: 55.845,
}

DEFAULT_COLOR: tuple[float, float, float] = (0.8, 0.0, 0.8)
DEFAULT_RADIUS: float = 0.08
