"""
./src/render_molecules/arrangement/scene_state.py

python -m src.render_molecules.arrangement.scene_state

This is not something that should be ran, instead it is just a container for all the bookkeeping things for all molecules.
It contains all the wrappers for everything to keep it all organized.
"""

import numpy as np


class MoleculeTemplate:
    """
    Immutable template for a single molecular species, such as cellobiose.
    It represents the chemical identity (what elements the molecule has) and topology in local coordinates (atoms' own positions in the molecule's own reference frame)

    This exists to create one unique Molecule Template per each molecule to avoid repetition and make life easier.
    It is also meant to be read-only
    """

    name: str  # Molecule name/label, e.g. cellobiose
    aids: np.ndarray  # Identifiers from final_aggregated.json -> not the IDX, but the ID each element is attributed
    elements: np.ndarray  # Atomic numbers aligned with aids by index
    local_xyz: tuple[
        np.ndarray, np.ndarray, np.ndarray
    ]  # The coordinates per each atom in the molecule's local space
    bonds_aid1: np.ndarray  # Starting Atom ID for bond
    bonds_aid2: np.ndarray  # Ending Atom ID for bond
    bond_order: np.ndarray  # Bond order


class MoleculeInstance:
    """
    The copy of the template in world space/environment
    Represents the spatial states and is linked to the molecule template

    Why it exists:
        - To allow me to place many copies without duplicating the template, only creating new instances.
        - Gives a clear distinction between molecule identity and location/position

    Use:
        - One object per molecule in the scene
        - Placement and rotation of instances
    """

    template_id: int  # Index/key pointing into SceneState.templates
    position: np.ndarray  # The world translation vector for the molecule instance
    rotation: np.ndarray  # World orientation transformation for the instance
    velocity: np.ndarray  # Not yet used, for future time-related shenanigans
    id: int  # Unique ID for bookkeeping


class ObjectState:
    """
    The full snapshot of a single object.
    """

    object_key: str  # Canonical unique key from JSON, e.g. books_2
    object_name: str  # Category/type name, e.g. books
    instance_id: str  # Runtime instance identifier; defaults to object_key
    display_name: str  # Human-readable label for logs/debug views
    templates: list[
        MoleculeTemplate
    ]  # The unique molecular templates, each having a unique IDX
    instances: list[
        MoleculeInstance
    ]  # All placed molecules currently in the scene. Points to an IDX in the templates
    box_bottom: np.ndarray  # BBox corner
    box_top: np.ndarray  # BBox corner
    rng_seed: int  # Random seed for reproducible arrangement


class Environment:
    scene_states: list[ObjectState]
