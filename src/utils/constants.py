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

ELEMENT_RADII: dict[int, float] = {  # In metres
    1: 1.20e-10,
    6: 1.70e-10,
    7: 1.55e-10,
    8: 1.52e-10,
    14: 2.10e-10,
    16: 1.80e-10,
    20: 2.31e-10,
    26: 1.94e-10,
}

ELEMENT_MASSES: dict[int, float] = {  # g/mol
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
DEFAULT_RADIUS: float = 1.40e-10  # In metres

SCENE_SCALE: float = 0.335 / 0.37790948  # Converts one unit in normalized capture to 1m

ANGSTROM_TO_METRES: float = 1e-10

# -------------------------
# Zoom transition constants
# -------------------------

LOG_ROOM_DISTANCE: float = 0.6  # log10 of the camera distance in m at room scale (~4m)
LOG_MOL_DISTANCE: float = (
    -8.3
)  # log10 of the camera distance in metres at molecular scale (~5nm)
SCROLL_STEP_SIZE: float = 0.15
BASE_MOVEMENT_SPEED: float = 1.5
ROOM_REFERENCE_DISTANCE: float = (
    4.0  # Camera distance in m at which BASE_MOVEMENT_SPEED applies
)
FADE_FOV_START: float = 10.0


# -------------------------
# Molecular Chunk Streaming
# -------------------------

CHUNK_SIZE_A: float = 100.0
LOAD_RADIUS_CHUNKS: int = 1
UNLOAD_RADIUS_CHUNKS: int = 2
MOL_CAM_SPEED_A: float = 50.0
MOL_VIEW_SCALE: float = 0.01
MAX_CHUNKS_PER_FRAME: int = 2
CHUNK_MOL_COUNT_PER_TEMPLATE: int = 2
WORLD_CHUNKS: int = 10  # World loops every WORLD_CHUNKS * CHUNK_SIZE_A Angstroms per axis

# ---------------------------------------------------------------------------
# Crystal structure element classification
# ---------------------------------------------------------------------------

METALLIC_ELEMENTS: frozenset[int] = frozenset({
    13, 20, 22, 24, 25, 26, 27, 28, 29, 30, 47, 79,
})
"""Atomic numbers of common metallic elements encountered in household objects."""
