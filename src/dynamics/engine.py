"""
./src/dynamics/engine.py

MACE model loading and force evaluation.
Routes organic molecules to MACE-OFF23 and metals to MACE-MP-0.
"""

from __future__ import annotations

import logging
import threading

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


def _patch_double_for_mps() -> None:
    """
    Monkey-patch torch.Tensor.double to return float() on MPS tensors.

    MACE's forward pass calls .double() in two places for numerical precision.
    MPS only supports float32, so those calls raise TypeError. This patch
    makes .double() a no-op (stays float32) for MPS tensors only.
    Accuracy is slightly reduced but acceptable for visualization.
    """
    import torch

    if getattr(torch.Tensor, "_mps_double_patched", False):
        return
    _orig = torch.Tensor.double

    def _mps_safe_double(self):
        if self.device.type == "mps":
            return self.float()
        return _orig(self)

    torch.Tensor.double = _mps_safe_double  # type: ignore[method-assign]
    torch.Tensor._mps_double_patched = True  # type: ignore[attr-defined]


def _load_and_cast(model_fn, device: str):
    """Load a MACE model, casting to float32 and moving to MPS if needed."""
    import torch

    if device == "mps":
        _patch_double_for_mps()
        load_device = "cpu"
        model = model_fn(load_device)
        model = model.float().to(device)
    else:
        model = model_fn(device)
    model.eval()
    return model


def load_mace_off23(device: str) -> torch.nn.Module:
    """
    Load the MACE-OFF23 model for organic molecules.

    Args:
        device: Torch device string ("mps", "cuda", "cpu").

    Returns:
        Loaded model in eval mode.
    """
    from mace.calculators import mace_off

    return _load_and_cast(
        lambda d: mace_off(model="medium", device=d, return_raw_model=True),
        device,
    )


def load_mace_mp0(device: str) -> torch.nn.Module:
    """
    Load the MACE-MP-0 universal model for metals and inorganics.

    Args:
        device: Torch device string.

    Returns:
        Loaded model in eval mode.
    """
    from mace.calculators import mace_mp

    return _load_and_cast(
        lambda d: mace_mp(model="medium", device=d, return_raw_model=True),
        device,
    )


def _build_mace_input(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    device: str,
    r_max: float,
    z_list: list[int],
) -> dict:
    """
    Build the input dict that MACE models expect.

    Args:
        positions: Shape (N, 3) in Angstroms.
        atomic_numbers: Shape (N,) of ints.
        device: Torch device string.
        r_max: Cutoff radius for neighbor list in Angstroms.
        z_list: Sorted list of atomic numbers the model was trained on.
                Used to build one-hot node_attrs of shape (N, n_species).

    Returns:
        Dict suitable for a MACE model forward pass.
    """
    import torch
    from scipy.spatial.distance import cdist

    # MPS only supports float32; use float64 everywhere else for numerical accuracy.
    fdtype = torch.float32 if device == "mps" else torch.float32

    n = len(positions)
    dists = cdist(positions, positions)
    edge_mask = (dists < r_max) & (dists > 1e-8)
    src, dst = np.where(edge_mask)

    pos_tensor = torch.tensor(positions, dtype=fdtype, device=device)
    edge_index = torch.tensor(
        np.stack([src, dst], axis=0) if len(src) else np.zeros((2, 0), dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    shifts = torch.zeros((len(src), 3), dtype=fdtype, device=device)
    batch = torch.zeros(n, dtype=torch.long, device=device)

    # Build one-hot node_attrs: shape (N, n_species)
    z_to_idx = {z: i for i, z in enumerate(z_list)}
    n_species = len(z_list)
    one_hot = np.zeros((n, n_species), dtype=np.float32)
    for i, z in enumerate(atomic_numbers):
        idx = z_to_idx.get(int(z))
        if idx is not None:
            one_hot[i, idx] = 1.0
    node_attrs = torch.tensor(one_hot, dtype=fdtype, device=device)

    return {
        "positions": pos_tensor.requires_grad_(True),
        "node_attrs": node_attrs,
        "edge_index": edge_index,
        "shifts": shifts,
        "batch": batch,
        "ptr": torch.tensor([0, n], dtype=torch.long, device=device),
        "cell": torch.zeros((1, 3, 3), dtype=fdtype, device=device),
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
    # Serializes MPS calls across threads — concurrent MPS gradient tapes OOM quickly
    _mps_lock: threading.Lock = threading.Lock()

    def __init__(self, is_metallic: bool = False) -> None:
        self.device: str = _select_device()
        self.is_metallic: bool = is_metallic
        self._model: torch.nn.Module | None = None
        self._r_max: float = 5.0
        self._z_list: list[int] = []

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
                logger.warning("Failed to load on %s, falling back to CPU", self.device)
                self.device = "cpu"
                if self.is_metallic:
                    self._model = load_mace_mp0("cpu")
                else:
                    self._model = load_mace_off23("cpu")
            else:
                raise

        if hasattr(self._model, "r_max"):
            self._r_max = float(self._model.r_max)
        if hasattr(self._model, "atomic_numbers"):
            self._z_list = self._model.atomic_numbers.tolist()

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
            pos_angstrom, atomic_numbers, self.device, self._r_max, self._z_list
        )

        import torch

        lock = MDEngine._mps_lock if self.device == "mps" else None
        if lock is not None:
            lock.acquire()
        try:
            with torch.enable_grad():
                output = self._model(inputs, training=False)

            energy_ev = float(output["energy"].detach().cpu().item())
            forces_ev_a = output["forces"].detach().cpu().double().numpy()
        finally:
            if lock is not None:
                lock.release()
                torch.mps.empty_cache()

        forces_n = forces_ev_a * (EV_TO_JOULE / ANGSTROM_TO_METRE)
        energy_j = energy_ev * EV_TO_JOULE

        return forces_n, energy_j
