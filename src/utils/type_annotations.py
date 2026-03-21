"""
./src/utils/typing.py

Contains all type annotations used throughout code for DRY principle.
"""

from typing import NotRequired, TypedDict, Optional

import torch

# ======= Annotations/Aggregations =======


class AnnotatedObjectDetails(TypedDict):
    name: str
    materials: list[str]
    corners: dict[str, list[list[float]]]
    base_normal: list[float]


Annotations = dict[str, AnnotatedObjectDetails]


class AtomDetails(TypedDict):
    idx: int
    symbol: str
    position: list[float]


class BondDetails(TypedDict):
    begin: int
    end: int
    type: float


class SimDetails(TypedDict):
    """RDKit-generated 3D structure (mol_details_rdkit.py)."""
    atoms: list[AtomDetails]
    bonds: list[BondDetails]


# ======= PubChem 3D structure =======

class PubChemAtoms(TypedDict):
    aid: list[int]      # atom IDs
    element: list[int]  # atomic numbers (parallel to aid)


class PubChemBonds(TypedDict):
    aid1: list[int]   # begin atom IDs
    aid2: list[int]   # end atom IDs
    order: list[int]  # bond orders (parallel to aid1/aid2)


class PubChemConformer(TypedDict):
    x: list[float]
    y: list[float]
    z: list[float]


class PubChemCoords(TypedDict):
    type: list[int]
    aid: list[int]
    conformers: list[PubChemConformer]


class PubChemSimDetails(TypedDict):
    """PubChem-sourced 3D structure (mol_details_pubchem.py)."""
    atoms: PubChemAtoms
    bonds: PubChemBonds
    coords: list[PubChemCoords]


class Molecule(TypedDict):
    formula: str
    smiles: str
    sim_details: NotRequired[PubChemSimDetails]


class AggregatedObjectDetails(AnnotatedObjectDetails):
    composition: dict[str, Molecule]


Aggregations = dict[str, AggregatedObjectDetails]

# ==== Legacy =====


class ObjectData(TypedDict):
    name: str
    materials: list[str]
    bounding_box: list[list[float]]


class Result(TypedDict):
    scores: torch.Tensor | list  # Shape: (N,) the number of objects
    boxes: torch.Tensor | list[list]  # Shape: (N, 4), where N is number of objects
    labels: list[str]


Detection = dict[str, dict[str, ObjectData]]
