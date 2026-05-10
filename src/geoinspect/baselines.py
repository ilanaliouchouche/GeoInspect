"""Baseline factories for Integrated Gradients on meshes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np

from .operators import (
    MeshOperators,
    coerce_operators,
    infer_num_vertices_from_mass,
    mass_to_vector,
    require_mass,
)
from .smoothing import smooth_map
from .types import MetadataDict, TensorLike

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class BaselineConfig:
    """Configuration for baseline generation."""

    kind: str = "zero"
    kwargs: MetadataDict = field(default_factory=dict)


def build_baseline(
    features: TensorLike,
    operators: MeshOperators,
    config: BaselineConfig,
) -> TensorLike:
    """Create a baseline tensor for Integrated Gradients.

    Supported baseline kinds:
    - ``zero``
    - ``mean``
    - ``heat``
    - ``spectral_lowpass``
    """
    ops = coerce_operators(operators)
    kind = config.kind.strip().lower()

    if kind == "zero":
        return _zeros_like(features)
    if kind == "mean":
        return _mass_mean_baseline(features, ops)
    if kind == "heat":
        tau = _optional_float(config.kwargs.get("tau", config.kwargs.get("t0", 1e-2)), default=1e-2)
        num_modes = _optional_positive_int(config.kwargs.get("num_modes"))
        return smooth_map(features, operators=ops, method="heat", tau=tau, num_modes=num_modes)
    if kind == "spectral_lowpass":
        num_modes_value = config.kwargs.get("num_modes", config.kwargs.get("K", 32))
        num_modes = _optional_positive_int(num_modes_value)
        if num_modes is None:
            num_modes = 32
        return smooth_map(features, operators=ops, method="heat", tau=0.0, num_modes=num_modes)

    raise ValueError("`baseline.kind` must be one of: zero, mean, heat, spectral_lowpass.")


def _zeros_like(value: TensorLike) -> TensorLike:
    if torch is not None and isinstance(value, torch.Tensor):
        return torch.zeros_like(value)
    return np.zeros_like(np.asarray(value))


def _mass_mean_baseline(features: TensorLike, operators: MeshOperators) -> TensorLike:
    mass = require_mass(operators)
    n_vertices = infer_num_vertices_from_mass(mass)
    vertex_axis = _vertex_axis_from_feature_shape(_shape_of(features), n_vertices)
    mass_vector = mass_to_vector(mass)

    if torch is not None and isinstance(features, torch.Tensor):
        if isinstance(mass_vector, torch.Tensor):
            mass_tensor = mass_vector.to(dtype=features.dtype, device=features.device)
        else:
            mass_tensor = torch.as_tensor(
                mass_vector,
                dtype=features.dtype,
                device=features.device,
            )

        reshape = [1] * features.ndim
        reshape[vertex_axis] = n_vertices
        weighted = features * mass_tensor.reshape(reshape)
        denominator = torch.sum(mass_tensor)
        if float(denominator.detach().cpu().item()) <= 0.0:
            raise ValueError("Mass vector must have strictly positive sum for mean baseline.")
        mean_feature = torch.sum(weighted, dim=vertex_axis, keepdim=True) / denominator
        return torch.ones_like(features) * mean_feature

    array = np.asarray(features)
    mass_arr = np.asarray(mass_vector, dtype=array.dtype)
    reshape = [1] * array.ndim
    reshape[vertex_axis] = n_vertices
    weighted = array * mass_arr.reshape(reshape)
    denominator_np = float(np.sum(mass_arr))
    if denominator_np <= 0.0:
        raise ValueError("Mass vector must have strictly positive sum for mean baseline.")
    mean_feature_np = np.sum(weighted, axis=vertex_axis, keepdims=True) / denominator_np
    return np.ones_like(array) * mean_feature_np


def _shape_of(value: TensorLike) -> tuple[int, ...]:
    if torch is not None and isinstance(value, torch.Tensor):
        return tuple(int(v) for v in value.shape)
    return tuple(int(v) for v in np.asarray(value).shape)


def _optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    number = int(cast(int, value))
    if number <= 0:
        raise ValueError("Expected a strictly positive integer.")
    return number


def _optional_float(value: object, default: float) -> float:
    if value is None:
        return default
    return float(cast(float | int | str, value))


def _vertex_axis_from_feature_shape(shape: tuple[int, ...], n_vertices: int) -> int:
    if len(shape) == 1:
        if shape[0] != n_vertices:
            raise ValueError(f"Expected shape [n] with n={n_vertices}, got {shape}.")
        return 0

    if len(shape) == 2:
        if shape[0] == n_vertices and shape[1] == n_vertices:
            return 0
        if shape[0] == n_vertices:
            return 0
        if shape[1] == n_vertices:
            return 1
        raise ValueError(f"Expected shape [n, C] with n={n_vertices}, got {shape}.")

    if len(shape) == 3:
        if shape[1] == n_vertices:
            return 1
        if shape[0] == n_vertices:
            return 0
        if shape[2] == n_vertices:
            return 2
        raise ValueError(f"Expected shape [B, n, C] with n={n_vertices}, got {shape}.")

    raise ValueError(
        f"Unsupported tensor rank {len(shape)} for shape {shape}. Expected [n,C] or [B,n,C]."
    )
