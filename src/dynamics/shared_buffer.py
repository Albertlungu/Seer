"""
./src/dynamics/shared_buffer.py

Double-buffered numpy array for lock-free position transfer
between the MD thread and the render thread.
"""

import threading

import numpy as np


class SharedPositionBuffer:
    """
    Two (N,3) float64 arrays with atomic index swap.
    The MD thread writes into one buffer, then swaps.
    The render thread reads the other buffer.

    Args:
        n_atoms: Number of atoms in the simulation.
    """

    def __init__(self, n_atoms: int) -> None:
        self._buffers: list[np.ndarray] = [
            np.zeros((n_atoms, 3), dtype=np.float64),
            np.zeros((n_atoms, 3), dtype=np.float64),
        ]
        self._write_idx: int = 0
        self._lock: threading.Lock = threading.Lock()

    def write(self, positions: np.ndarray) -> None:
        """
        Copy positions into the write buffer, then swap indices.
        Called from the MD thread.

        Args:
            positions: Array of shape (N, 3) with updated atom positions.
        """
        np.copyto(self._buffers[self._write_idx], positions)
        with self._lock:
            self._write_idx = 1 - self._write_idx

    def read(self) -> np.ndarray:
        """
        Return a view of the current read buffer.
        Called from the render thread.

        Returns:
            Array of shape (N, 3). Do not hold this reference across frames.
        """
        with self._lock:
            read_idx = 1 - self._write_idx
        return self._buffers[read_idx]

    @property
    def n_atoms(self) -> int:
        return self._buffers[0].shape[0]
