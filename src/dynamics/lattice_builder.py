"""
./src/dynamics/lattice_builder.py

Generates crystal lattice atom positions for metallic elements.
"""

import logging
from dataclasses import dataclass

import numpy as np

from src.dynamics.constants import CRYSTAL_STRUCTURES, DEFAULT_CRYSTAL_TYPE
from src.utils.constants import ELEMENT_RADII

logger = logging.getLogger(__name__)


@dataclass
class CrystalStructure:
    structure_type: str
    lattice_parameter: float  # metres
    basis_positions: np.ndarray  # shape (B, 3), fractional coordinates


def get_crystal_structure(atomic_number: int) -> CrystalStructure:
    """
    Look up the crystal structure for a metallic element.

    Args:
        atomic_number: Element number.

    Returns:
        CrystalStructure with lattice type, parameter, and basis.
    """
    if atomic_number in CRYSTAL_STRUCTURES:
        stype, a, basis = CRYSTAL_STRUCTURES[atomic_number]
        return CrystalStructure(
            structure_type=stype,
            lattice_parameter=a,
            basis_positions=np.array(basis, dtype=np.float32),
        )

    radius = ELEMENT_RADII.get(atomic_number, 1.5e-10)
    a_est = 2.0 * radius * np.sqrt(2)
    logger.warning(
        "No crystal data for Z=%d, using FCC with a=%.3e m", atomic_number, a_est
    )
    return CrystalStructure(
        structure_type=DEFAULT_CRYSTAL_TYPE,
        lattice_parameter=a_est,
        basis_positions=np.array(
            [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
            dtype=np.float32,
        ),
    )


def build_lattice(
    structure: CrystalStructure,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> np.ndarray:
    """
    Tile the unit cell to fill a bounding box.

    Args:
        structure: Crystal structure to tile.
        bbox_min: Lower corner of the bounding box in metres, shape (3,).
        bbox_max: Upper corner of the bounding box in metres, shape (3,).

    Returns:
        Atom positions of shape (M, 3) in metres.
    """
    a = structure.lattice_parameter
    dims = bbox_max - bbox_min
    n_cells = np.ceil(dims / a).astype(int)

    max_atoms = int(np.prod(n_cells)) * len(structure.basis_positions)
    positions = np.empty((max_atoms, 3), dtype=np.float64)
    count = 0

    for ix in range(int(n_cells[0])):
        for iy in range(int(n_cells[1])):
            for iz in range(int(n_cells[2])):
                cell_origin = bbox_min + np.array([ix, iy, iz], dtype=np.float64) * a
                for frac in structure.basis_positions:
                    pos = cell_origin + frac * a
                    if np.all(pos >= bbox_min) and np.all(pos <= bbox_max):
                        positions[count] = pos
                        count += 1

    return positions[:count].copy()


def is_metallic(object_data: dict) -> bool:
    """
    Determine if an object should be treated as a metallic lattice.

    Heuristic: if none of the composition entries have sim_details (PubChem data),
    the object is metallic.

    Args:
        object_data: One object entry from the aggregated JSON.

    Returns:
        True if the object is metallic.
    """
    composition = object_data.get("composition", {})
    if not composition:
        return True
    return all("sim_details" not in mol for mol in composition.values())
