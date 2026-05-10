"""Normalization utilities for attribution maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operators import (
    MeshOperators,
    coerce_operators,
    infer_num_vertices_from_mass,
    infer_vertex_axis,
    is_torch_tensor,
    mass_to_vector,
    require_mass,
)
from .types import TensorLike

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class NormalizationConfig:
    """Normalization strategy for scalar maps."""

    method: str | None = None


def normalize_map(
    scalar_map: TensorLike,
    operators: MeshOperators,
    method: str | None = None,
) -> TensorLike:
    """Normalize an attribution map.

    Supported methods:
    - ``None`` / ``"none"``
    - ``"minmax"``
    - ``"max_abs"``
    - ``"area_integral"``
    """
    ops = coerce_operators(operators)
    method_name = _canonical_method_name(method)

    if is_torch_tensor(scalar_map):
        return _normalize_torch(scalar_map, ops, method_name)
    return _normalize_numpy(scalar_map, ops, method_name)


def _canonical_method_name(method: str | None) -> str:
    if method is None:
        return "none"
    method_name = method.strip().lower()
    if method_name in {"", "none"}:
        return "none"
    if method_name not in {"minmax", "max_abs", "area_integral"}:
        raise ValueError("`method` must be one of: none, minmax, max_abs, area_integral.")
    return method_name


def _normalize_torch(
    scalar_map: TensorLike,
    operators: MeshOperators,
    method: str,
) -> TensorLike:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Torch backend unavailable.")

    tensor = scalar_map.clone()
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)

    if method == "none":
        return tensor

    eps = torch.finfo(tensor.dtype).eps

    if method == "minmax":
        value_min = torch.min(tensor)
        value_max = torch.max(tensor)
        span = value_max - value_min
        if torch.abs(span) <= eps:
            return torch.zeros_like(tensor)
        return (tensor - value_min) / span

    if method == "max_abs":
        value_max_abs = torch.max(torch.abs(tensor))
        if torch.abs(value_max_abs) <= eps:
            return torch.zeros_like(tensor)
        return tensor / value_max_abs

    mass = require_mass(operators)
    mass_vector = mass_to_vector(mass)
    if not is_torch_tensor(mass_vector):
        mass_vector = torch.as_tensor(
            mass_vector,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        mass_vector = mass_vector.to(dtype=tensor.dtype, device=tensor.device)

    n_vertices = infer_num_vertices_from_mass(mass)
    axis = infer_vertex_axis(tuple(tensor.shape), n_vertices)
    reshape = [1] * tensor.ndim
    reshape[axis] = n_vertices
    weighted = tensor * mass_vector.reshape(reshape)
    denominator = torch.sum(weighted, dim=axis, keepdim=True)

    if torch.any(torch.abs(denominator) <= eps):
        raise ValueError("Area-integral normalization is undefined for zero integral maps.")

    return tensor / denominator


def _normalize_numpy(
    scalar_map: TensorLike,
    operators: MeshOperators,
    method: str,
) -> TensorLike:
    array = np.asarray(scalar_map)
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64)
    else:
        array = array.copy()

    if method == "none":
        return array

    eps = np.finfo(array.dtype).eps

    if method == "minmax":
        value_min = float(np.min(array))
        value_max = float(np.max(array))
        span = value_max - value_min
        if abs(span) <= eps:
            return np.zeros_like(array)
        return (array - value_min) / span

    if method == "max_abs":
        value_max_abs = float(np.max(np.abs(array)))
        if abs(value_max_abs) <= eps:
            return np.zeros_like(array)
        return array / value_max_abs

    mass = require_mass(operators)
    mass_vector = np.asarray(mass_to_vector(mass), dtype=array.dtype)
    n_vertices = infer_num_vertices_from_mass(mass)
    axis = infer_vertex_axis(array.shape, n_vertices)
    reshape = [1] * array.ndim
    reshape[axis] = n_vertices

    weighted = array * mass_vector.reshape(reshape)
    denominator = np.sum(weighted, axis=axis, keepdims=True)
    if np.any(np.abs(denominator) <= eps):
        raise ValueError("Area-integral normalization is undefined for zero integral maps.")

    return array / denominator
