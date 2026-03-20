"""
./src/utils/typing.py

Contains all type annotations used throughout code for DRY principle.
"""

from typing import NotRequired, TypedDict

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
    atoms: list[AtomDetails]
    bonds: list[BondDetails]


class Molecule(TypedDict):
    formula: str
    smiles: str
    sim_details: NotRequired[SimDetails]


class AggregatedObjectDetails(AnnotatedObjectDetails, total=False):
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
