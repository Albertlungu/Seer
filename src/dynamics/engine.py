"""
./src/dynamics/engine.py

MACE model loading and force evaluation.
Routes organic molecules to MACE-OFF23 and metals to MACE-MP-0.
"""

from __future__ import annotations

import logging

import numpy as np

from src.dynamics.constants import ANGSTROM_TO_METRE, EV_TO_JOULE

logger = logging.getLogger(__name__)


def _select_device() -> str:
    """Pick the best available torch device."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_mace_off23(device: str) -> torch.nn.Module:
    """
    Load the MACE-OFF23 model for organic molecules.

    Args:
        device: Torch device string ("mps", "cuda", "cpu").

    Returns:
        Loaded model in eval mode.
    """
    from mace.calculators import mace_off

    model = mace_off(model="medium", device=device, return_raw_model=True)
    model.eval()
    return model


def load_mace_mp0(device: str) -> torch.nn.Module:
    """
    Load the MACE-MP-0 universal model for metals and inorganics.

    Args:
        device: Torch device string.

    Returns:
        Loaded model in eval mode.
    """
    from mace.calculators import mace_mp

    model = mace_mp(model="medium", device=device, return_raw_model=True)
    model.eval()
    return model


def _build_mace_input(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    device: str,
    r_max: float,
) -> dict:
    """
    Build the input dict that MACE models expect.

    Args:
        positions: Shape (N, 3) in Angstroms.
        atomic_numbers: Shape (N,) of ints.
        device: Torch device string.
        r_max: Cutoff radius for neighbor list in Angstroms.

    Returns:
        Dict with positions, atomic_numbers, edge_index, shifts tensors.
    """
    import torch
    from scipy.spatial.distance import cdist

    n = len(positions)
    dists = cdist(positions, positions)
    edge_mask = (dists < r_max) & (dists > 1e-8)
    src, dst = np.where(edge_mask)

    pos_tensor = torch.tensor(positions, dtype=torch.float64, device=device)
    z_tensor = torch.tensor(atomic_numbers, dtype=torch.long, device=device)
    edge_index = torch.tensor(
        np.stack([src, dst], axis=0), dtype=torch.long, device=device
    )
    shifts = torch.zeros((len(src), 3), dtype=torch.float64, device=device)
    batch = torch.zeros(n, dtype=torch.long, device=device)

    return {
        "positions": pos_tensor.requires_grad_(True),
        "node_attrs": z_tensor,
        "edge_index": edge_index,
        "shifts": shifts,
        "batch": batch,
        "ptr": torch.tensor([0, n], dtype=torch.long, device=device),
        # Non-periodic system: zero cell, no pbc
        "cell": torch.zeros((1, 3, 3), dtype=torch.float64, device=device),
        "pbc": torch.zeros(3, dtype=torch.bool, device=device),
    }


class MDEngine:
    """
    Wraps MACE model loading and force evaluation.

    The engine holds two models: MACE-OFF23 for organic atoms and MACE-MP-0 for
    metallic/inorganic atoms. Material routing is determined at initialization
    by the `is_metallic` flag.

    Args:
        is_metallic: If True, uses MACE-MP-0. If False, uses MACE-OFF23.
    """

    _OFF23_ELEMENTS: frozenset[int] = frozenset({1, 6, 7, 8, 9, 16, 17})

    def __init__(self, is_metallic: bool = False) -> None:
        self.device: str = _select_device()
        self.is_metallic: bool = is_metallic
        self._model: torch.nn.Module | None = None
        self._r_max: float = 5.0

    def load_model(self) -> None:
        """Load the appropriate MACE model. Call once before evaluate_forces."""
        try:
            if self.is_metallic:
                self._model = load_mace_mp0(self.device)
                logger.info("Loaded MACE-MP-0 on %s", self.device)
            else:
                self._model = load_mace_off23(self.device)
                logger.info("Loaded MACE-OFF23 on %s", self.device)
        except Exception:
            if self.device != "cpu":
                logger.warning(
                    "Failed to load on %s, falling back to CPU", self.device
                )
                self.device = "cpu"
                if self.is_metallic:
                    self._model = load_mace_mp0("cpu")
                else:
                    self._model = load_mace_off23("cpu")
            else:
                raise

        if hasattr(self._model, "r_max"):
            self._r_max = float(self._model.r_max)

    def evaluate_forces(
        self,
        positions: np.ndarray,
        atomic_numbers: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Run one forward pass and compute forces via autograd.

        Args:
            positions: Shape (N, 3) in metres.
            atomic_numbers: Shape (N,) element numbers.

        Returns:
            Tuple of (forces in Newtons shape (N,3), total energy in Joules).

        Raises:
            RuntimeError: If model has not been loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        pos_angstrom = positions / ANGSTROM_TO_METRE

        inputs = _build_mace_input(
            pos_angstrom, atomic_numbers, self.device, self._r_max
        )

        import torch

        with torch.enable_grad():
            output = self._model(inputs, training=False)

        energy_ev = float(output["energy"].detach().cpu().item())
        forces_ev_a = output["forces"].detach().cpu().numpy()

        forces_n = forces_ev_a * (EV_TO_JOULE / ANGSTROM_TO_METRE)
        energy_j = energy_ev * EV_TO_JOULE

        return forces_n, energy_j
