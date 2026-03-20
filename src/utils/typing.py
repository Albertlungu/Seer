"""
./src/utils/typing.py

Contains all type annotations used throughout code for DRY principle.
"""

from typing import TypedDict

# ======= Annotations/Aggregations =======


class AnnotatedObjectDetails(TypedDict):
    name: str
    materials: list[str]
    corners: dict[str, list[list[float]]]
    base_normal: list[float]


Annotations = dict[str, AnnotatedObjectDetails]


class AggregatedObjectDetails(AnnotatedObjectDetails, total=False):
    composition: dict[str, dict]


Aggregations = dict[str, AggregatedObjectDetails]

#


class ObjectData(TypedDict):
    name: str
    materials: list[str]
    bounding_box: list[list[float]]


class Result(TypedDict):
    scores: torch.Tensor | list  # Shape: (N,) the number of objects
    boxes: torch.Tensor | list[list]  # Shape: (N, 4), where N is number of objects
    labels: list[str]


Detection = dict[str, dict[str, ObjectData]]
