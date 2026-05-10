"""Mesh-aware smoothing utilities for attribution maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from .operators import (
    MeshOperators,
    coerce_operators,
    infer_num_vertices_from_mass,
    infer_vertex_axis,
    is_scipy_sparse,
    is_torch_tensor,
    mass_to_vector,
    require_mass,
    require_spectral_operators,
    require_stiffness_or_laplacian,
)
from .types import TensorLike

try:
    import scipy.sparse as sp
    from scipy.sparse.linalg import factorized
except ImportError:  # pragma: no cover - optional dependency
    sp = None
    factorized = None

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class SmoothingConfig:
    """Post-processing parameters for map smoothing."""

    method: str | None = None
    tau: float | None = None
    num_modes: int | None = None


def smooth_map(
    scalar_map: TensorLike,
    operators: MeshOperators,
    method: str = "helmholtz",
    tau: float | None = None,
    num_modes: int | None = None,
) -> TensorLike:
    """Apply optional smoothing to a vertex-wise scalar map.

    Supported methods:
    - ``none``
    - ``helmholtz``
    - ``heat``
    """
    ops = coerce_operators(operators)
    method_name = _canonical_method_name(method)

    if method_name == "none":
        return _clone_like(scalar_map)

    if tau is None:
        tau_value = 1e-2
    else:
        tau_value = float(tau)

    if tau_value < 0.0:
        raise ValueError("`tau` must be non-negative.")

    if num_modes is not None and num_modes <= 0:
        raise ValueError("`num_modes` must be a positive integer.")

    if method_name == "helmholtz":
        return _helmholtz_smoothing(scalar_map, ops, tau_value)

    return _heat_smoothing(scalar_map, ops, tau_value, num_modes)


def _canonical_method_name(method: str | None) -> str:
    if method is None:
        return "none"
    method_name = method.strip().lower()
    if method_name in {"", "none"}:
        return "none"
    if method_name not in {"helmholtz", "heat"}:
        raise ValueError("`method` must be one of: none, helmholtz, heat.")
    return method_name


def _clone_like(value: TensorLike) -> TensorLike:
    if is_torch_tensor(value):
        return value.clone()
    return np.asarray(value).copy()


def _heat_smoothing(
    scalar_map: TensorLike,
    operators: MeshOperators,
    tau: float,
    num_modes: int | None,
) -> TensorLike:
    evals, evecs = require_spectral_operators(operators)
    mass = require_mass(operators)

    if is_torch_tensor(scalar_map):
        return _heat_torch(scalar_map, mass, evals, evecs, tau, num_modes)
    return _heat_numpy(scalar_map, mass, evals, evecs, tau, num_modes)


def _heat_numpy(
    scalar_map: TensorLike,
    mass: TensorLike,
    evals: TensorLike,
    evecs: TensorLike,
    tau: float,
    num_modes: int | None,
) -> np.ndarray:
    signal = np.asarray(scalar_map)
    if not np.issubdtype(signal.dtype, np.floating):
        signal = signal.astype(np.float64)
    else:
        signal = signal.copy()

    mass_vector = np.asarray(mass_to_vector(mass), dtype=signal.dtype)
    evals_array = np.asarray(evals, dtype=signal.dtype).reshape(-1)
    evecs_array = np.asarray(evecs, dtype=signal.dtype)

    n_vertices = infer_num_vertices_from_mass(mass)
    if evecs_array.ndim != 2 or evecs_array.shape[0] != n_vertices:
        raise ValueError("`evecs` must have shape [n_vertices, k].")

    axis = infer_vertex_axis(signal.shape, n_vertices)
    signal_vfirst = np.moveaxis(signal, axis, 0)
    flat_signal = signal_vfirst.reshape(n_vertices, -1)

    max_modes = min(evecs_array.shape[1], evals_array.shape[0])
    k_modes = max_modes if num_modes is None else min(num_modes, max_modes)
    if k_modes <= 0:
        raise ValueError("No spectral modes available for heat smoothing.")

    phi = evecs_array[:, :k_modes]
    lambdas = evals_array[:k_modes]
    heat_weights = np.exp(-tau * lambdas)

    projected = phi.T @ (flat_signal * mass_vector[:, None])
    smoothed_flat = phi @ (heat_weights[:, None] * projected)

    smoothed_vfirst = smoothed_flat.reshape(signal_vfirst.shape)
    return np.moveaxis(smoothed_vfirst, 0, axis)


def _heat_torch(
    scalar_map: TensorLike,
    mass: TensorLike,
    evals: TensorLike,
    evecs: TensorLike,
    tau: float,
    num_modes: int | None,
) -> TensorLike:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Torch backend unavailable.")

    signal = scalar_map.clone()
    if not signal.is_floating_point():
        signal = signal.to(dtype=torch.float32)

    if is_torch_tensor(mass):
        mass_vector = mass_to_vector(mass).to(dtype=signal.dtype, device=signal.device)
    else:
        mass_vector = torch.as_tensor(
            mass_to_vector(mass),
            dtype=signal.dtype,
            device=signal.device,
        )

    if is_torch_tensor(evals):
        evals_tensor = evals.to(dtype=signal.dtype, device=signal.device).reshape(-1)
    else:
        evals_tensor = torch.as_tensor(evals, dtype=signal.dtype, device=signal.device).reshape(-1)

    if is_torch_tensor(evecs):
        evecs_tensor = evecs.to(dtype=signal.dtype, device=signal.device)
    else:
        evecs_tensor = torch.as_tensor(evecs, dtype=signal.dtype, device=signal.device)

    n_vertices = infer_num_vertices_from_mass(mass)
    if evecs_tensor.ndim != 2 or evecs_tensor.shape[0] != n_vertices:
        raise ValueError("`evecs` must have shape [n_vertices, k].")

    axis = infer_vertex_axis(tuple(signal.shape), n_vertices)
    signal_vfirst = torch.movedim(signal, axis, 0)
    flat_signal = signal_vfirst.reshape(n_vertices, -1)

    max_modes = min(evecs_tensor.shape[1], evals_tensor.shape[0])
    k_modes = max_modes if num_modes is None else min(num_modes, max_modes)
    if k_modes <= 0:
        raise ValueError("No spectral modes available for heat smoothing.")

    phi = evecs_tensor[:, :k_modes]
    lambdas = evals_tensor[:k_modes]
    heat_weights = torch.exp(-tau * lambdas)

    projected = phi.transpose(0, 1) @ (flat_signal * mass_vector.reshape(n_vertices, 1))
    smoothed_flat = phi @ (heat_weights.reshape(k_modes, 1) * projected)

    smoothed_vfirst = smoothed_flat.reshape(signal_vfirst.shape)
    return torch.movedim(smoothed_vfirst, 0, axis)


def _helmholtz_smoothing(
    scalar_map: TensorLike,
    operators: MeshOperators,
    tau: float,
) -> TensorLike:
    reference = scalar_map
    signal_np = _to_numpy_array(scalar_map)
    smoothed_np = _helmholtz_numpy(signal_np, operators, tau)

    if is_torch_tensor(reference):
        if torch is None:  # pragma: no cover
            raise RuntimeError("Torch backend unavailable.")
        return torch.as_tensor(smoothed_np, dtype=reference.dtype, device=reference.device)
    return smoothed_np


def _helmholtz_numpy(
    scalar_map: np.ndarray,
    operators: MeshOperators,
    tau: float,
) -> np.ndarray:
    mass = require_mass(operators)
    stiffness, laplacian = require_stiffness_or_laplacian(operators)

    signal = scalar_map
    if not np.issubdtype(signal.dtype, np.floating):
        signal = signal.astype(np.float64)
    else:
        signal = signal.copy()

    n_vertices = infer_num_vertices_from_mass(mass)
    axis = infer_vertex_axis(signal.shape, n_vertices)

    signal_vfirst = np.moveaxis(signal, axis, 0)
    flat_signal = signal_vfirst.reshape(n_vertices, -1)

    mass_vector = np.asarray(mass_to_vector(mass), dtype=signal.dtype)
    mass_matrix = _to_sparse_matrix(mass, signal.dtype)
    rhs = mass_matrix @ flat_signal

    if stiffness is not None:
        stiffness_matrix = _to_sparse_matrix(stiffness, signal.dtype)
    else:
        assert laplacian is not None
        laplacian_matrix = _to_sparse_matrix(laplacian, signal.dtype)
        if sp is None:
            stiffness_matrix = np.diag(mass_vector) @ laplacian_matrix
        else:
            stiffness_matrix = sp.diags(mass_vector) @ laplacian_matrix

    if sp is not None:
        system_matrix_any = cast(Any, mass_matrix) + tau * cast(Any, stiffness_matrix)
        system_matrix = cast(Any, system_matrix_any.tocsc())
        solution = _solve_sparse_columns(system_matrix, rhs, operators, tau)
    else:  # pragma: no cover - SciPy available in test env
        system_matrix_dense = np.asarray(cast(Any, mass_matrix) + tau * cast(Any, stiffness_matrix))
        solution = np.linalg.solve(system_matrix_dense, rhs)

    smoothed_vfirst = solution.reshape(signal_vfirst.shape)
    return np.moveaxis(smoothed_vfirst, 0, axis)


def _solve_sparse_columns(
    system_matrix: object,
    rhs: np.ndarray,
    operators: MeshOperators,
    tau: float,
) -> np.ndarray:
    if factorized is None:  # pragma: no cover
        raise RuntimeError("SciPy sparse solver backend unavailable.")

    matrix = cast(Any, system_matrix)
    cache_key = f"helmholtz::{tau:.12g}::{matrix.shape[0]}::{matrix.nnz}"
    if cache_key not in operators._solver_cache:
        operators._solver_cache[cache_key] = factorized(matrix)

    solve_fn = operators._solver_cache[cache_key]
    if rhs.ndim == 1:
        return np.asarray(solve_fn(rhs), dtype=rhs.dtype)

    solutions = [
        np.asarray(solve_fn(rhs[:, col_idx]), dtype=rhs.dtype) for col_idx in range(rhs.shape[1])
    ]
    return np.column_stack(solutions)


def _to_numpy_array(value: TensorLike) -> np.ndarray:
    if is_torch_tensor(value):
        return cast(np.ndarray, value.detach().cpu().numpy())
    return np.asarray(value)


def _to_sparse_matrix(value: TensorLike, dtype: np.dtype[Any]) -> object:
    if sp is None:
        return np.asarray(_to_numpy_array(value), dtype=dtype)

    if is_torch_tensor(value):
        value_np = value.detach().cpu().numpy()
        if value_np.ndim == 1:
            return sp.diags(np.asarray(value_np, dtype=dtype)).tocsc()
        return sp.csc_matrix(np.asarray(value_np, dtype=dtype))

    if is_scipy_sparse(value):
        return value.astype(dtype).tocsc()

    array = np.asarray(value)
    if array.ndim == 1:
        return sp.diags(np.asarray(array, dtype=dtype)).tocsc()
    if array.ndim == 2:
        return sp.csc_matrix(np.asarray(array, dtype=dtype))
    raise ValueError("Operator must be a vector [n] or matrix [n, n].")
