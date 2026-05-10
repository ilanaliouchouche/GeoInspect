"""Mesh operator containers and validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from .types import TensorLike

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover - optional dependency
    sp = None

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "mass": ("mass", "massvec", "M"),
    "stiffness": ("stiffness", "S"),
    "laplacian": ("laplacian", "L"),
    "evals": ("evals", "eigenvalues", "Lambda"),
    "evecs": ("evecs", "eigenvectors", "Phi"),
    "grad_x": ("grad_x", "gradX"),
    "grad_y": ("grad_y", "gradY"),
}


@dataclass(slots=True)
class MeshOperators:
    """Precomputed geometric operators required by GeoInspect.

    The class accepts operators directly, but can also be constructed from
    DiffusionNet-like dicts / objects via :meth:`from_any`.
    """

    mass: TensorLike | None
    stiffness: TensorLike | None = None
    laplacian: TensorLike | None = None
    evals: TensorLike | None = None
    evecs: TensorLike | None = None
    grad_x: TensorLike | None = None
    grad_y: TensorLike | None = None
    _solver_cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_any(cls, source: TensorLike) -> MeshOperators:
        """Build ``MeshOperators`` from a dict/object or return as-is."""
        if isinstance(source, MeshOperators):
            return source

        if isinstance(source, Mapping):
            return cls(
                mass=_pick_from_mapping(source, _ALIAS_MAP["mass"]),
                stiffness=_pick_from_mapping(source, _ALIAS_MAP["stiffness"]),
                laplacian=_pick_from_mapping(source, _ALIAS_MAP["laplacian"]),
                evals=_pick_from_mapping(source, _ALIAS_MAP["evals"]),
                evecs=_pick_from_mapping(source, _ALIAS_MAP["evecs"]),
                grad_x=_pick_from_mapping(source, _ALIAS_MAP["grad_x"]),
                grad_y=_pick_from_mapping(source, _ALIAS_MAP["grad_y"]),
            )

        return cls(
            mass=_pick_from_object(source, _ALIAS_MAP["mass"]),
            stiffness=_pick_from_object(source, _ALIAS_MAP["stiffness"]),
            laplacian=_pick_from_object(source, _ALIAS_MAP["laplacian"]),
            evals=_pick_from_object(source, _ALIAS_MAP["evals"]),
            evecs=_pick_from_object(source, _ALIAS_MAP["evecs"]),
            grad_x=_pick_from_object(source, _ALIAS_MAP["grad_x"]),
            grad_y=_pick_from_object(source, _ALIAS_MAP["grad_y"]),
        )

    def num_vertices(self) -> int:
        """Infer vertex count from the mass operator."""
        mass = require_mass(self)
        return infer_num_vertices_from_mass(mass)


class MissingOperatorError(ValueError):
    """Raised when a required geometric operator is missing."""


def coerce_operators(operators: TensorLike) -> MeshOperators:
    """Return a ``MeshOperators`` instance from various input formats."""
    return MeshOperators.from_any(operators)


def require_mass(operators: MeshOperators) -> TensorLike:
    """Return the mass operator or raise a descriptive error."""
    if operators.mass is None:
        raise MissingOperatorError("`operators.mass` is required for mesh-aware attribution.")
    return operators.mass


def require_stiffness_or_laplacian(
    operators: MeshOperators,
) -> tuple[TensorLike | None, TensorLike | None]:
    """Return available stiffness / laplacian operator, ensuring at least one exists."""
    if operators.stiffness is None and operators.laplacian is None:
        raise MissingOperatorError(
            "Either `operators.stiffness` or `operators.laplacian` is required."
        )
    return operators.stiffness, operators.laplacian


def require_spectral_operators(operators: MeshOperators) -> tuple[TensorLike, TensorLike]:
    """Return eigenpairs required for spectral operations."""
    if operators.evals is None or operators.evecs is None:
        raise MissingOperatorError("`operators.evals` and `operators.evecs` are required.")
    return operators.evals, operators.evecs


def infer_num_vertices_from_mass(mass: TensorLike) -> int:
    """Infer number of vertices from vector/matrix mass representation."""
    if torch is not None and isinstance(mass, torch.Tensor):
        if mass.ndim == 1:
            return int(mass.shape[0])
        if mass.ndim == 2 and mass.shape[0] == mass.shape[1]:
            return int(mass.shape[0])
        raise ValueError("Mass tensor must be shape [n] or [n, n].")

    if sp is not None and sp.issparse(mass):
        if mass.shape[0] != mass.shape[1]:
            raise ValueError("Sparse mass matrix must be square.")
        return int(mass.shape[0])

    mass_arr = np.asarray(mass)
    if mass_arr.ndim == 1:
        return int(mass_arr.shape[0])
    if mass_arr.ndim == 2 and mass_arr.shape[0] == mass_arr.shape[1]:
        return int(mass_arr.shape[0])
    raise ValueError("Mass array must be shape [n] or [n, n].")


def mass_to_vector(mass: TensorLike) -> TensorLike:
    """Return lumped mass vector from vector or matrix representation."""
    if torch is not None and isinstance(mass, torch.Tensor):
        if mass.ndim == 1:
            return mass
        if mass.ndim == 2 and mass.shape[0] == mass.shape[1]:
            return torch.diagonal(mass)
        raise ValueError("Mass tensor must be shape [n] or [n, n].")

    if sp is not None and sp.issparse(mass):
        if mass.shape[0] != mass.shape[1]:
            raise ValueError("Sparse mass matrix must be square.")
        return mass.diagonal()

    mass_arr = np.asarray(mass)
    if mass_arr.ndim == 1:
        return mass_arr
    if mass_arr.ndim == 2 and mass_arr.shape[0] == mass_arr.shape[1]:
        return np.diag(mass_arr)
    raise ValueError("Mass array must be shape [n] or [n, n].")


def infer_vertex_axis(shape: tuple[int, ...], n_vertices: int) -> int:
    """Infer which axis corresponds to vertices for a map-like tensor."""
    if n_vertices <= 0:
        raise ValueError("`n_vertices` must be positive.")
    if not shape:
        raise ValueError("Input map must have at least one dimension.")

    matches = [idx for idx, dim in enumerate(shape) if dim == n_vertices]
    if not matches:
        raise ValueError(f"Could not find vertex axis of size {n_vertices} in shape {shape}.")
    if len(matches) == 1:
        return matches[0]
    if shape[-1] == n_vertices:
        return len(shape) - 1
    if shape[0] == n_vertices:
        return 0
    raise ValueError(f"Ambiguous vertex axis for shape {shape}; explicitly reshape input map.")


def is_torch_tensor(value: object) -> bool:
    """Return whether ``value`` is a torch tensor."""
    return torch is not None and isinstance(value, torch.Tensor)


def is_scipy_sparse(value: object) -> bool:
    """Return whether ``value`` is a SciPy sparse matrix/array."""
    return sp is not None and sp.issparse(value)


def _pick_from_mapping(
    source: Mapping[str, object],
    aliases: tuple[str, ...],
) -> object | None:
    for name in aliases:
        if name in source:
            return source[name]
    return None


def _pick_from_object(source: object, aliases: tuple[str, ...]) -> object | None:
    for name in aliases:
        if hasattr(source, name):
            return cast(object, getattr(source, name))
    return None
