"""Scientific sanity checks for mesh-aware XAI outputs."""

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
    require_stiffness_or_laplacian,
)
from .types import TensorLike

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover - optional dependency
    sp = None


@dataclass(slots=True)
class CheckResult:
    """Structured result for a scientific check."""

    name: str
    value: float
    passed: bool


def check_mass_consistency(operators: MeshOperators) -> CheckResult:
    """Verify mass positivity and finite total area."""
    ops = coerce_operators(operators)
    mass = require_mass(ops)
    mass_vector = np.asarray(_to_numpy(mass_to_vector(mass)), dtype=np.float64).reshape(-1)

    if mass_vector.size == 0:
        raise ValueError("Mass vector cannot be empty.")

    total_mass = float(np.sum(mass_vector))
    has_positive_entries = bool(np.all(mass_vector > 0.0))
    is_finite = bool(np.isfinite(total_mass) and np.all(np.isfinite(mass_vector)))
    passed = has_positive_entries and is_finite and total_mass > 0.0

    return CheckResult(
        name="mass_consistency_total_area",
        value=total_mass,
        passed=passed,
    )


def check_constant_signal(operators: MeshOperators) -> CheckResult:
    """Verify that a constant signal has near-zero stiffness/Laplacian response."""
    ops = coerce_operators(operators)
    mass = require_mass(ops)
    stiffness, laplacian = require_stiffness_or_laplacian(ops)

    n_vertices = infer_num_vertices_from_mass(mass)
    ones = np.ones((n_vertices,), dtype=np.float64)

    operator = stiffness if stiffness is not None else laplacian
    if operator is None:  # pragma: no cover - unreachable after require_stiffness_or_laplacian
        raise ValueError("Missing stiffness/laplacian operator.")

    residual = _matmul_operator(operator, ones)
    residual_norm = float(np.linalg.norm(residual))
    normalized_residual = residual_norm / max(float(np.sqrt(n_vertices)), 1.0)

    return CheckResult(
        name="constant_signal_response",
        value=normalized_residual,
        passed=normalized_residual <= 1e-6,
    )


def check_ig_completeness(
    density_map: TensorLike,
    operators: MeshOperators,
    delta_score: float,
) -> CheckResult:
    """Verify Integrated Gradients completeness error."""
    ops = coerce_operators(operators)
    mass = require_mass(ops)
    mass_vector = np.asarray(_to_numpy(mass_to_vector(mass)), dtype=np.float64).reshape(-1)
    n_vertices = infer_num_vertices_from_mass(mass)

    density = np.asarray(_to_numpy(density_map), dtype=np.float64)
    vertex_axis = infer_vertex_axis(density.shape, n_vertices)
    reshape = [1] * density.ndim
    reshape[vertex_axis] = n_vertices

    weighted = density * mass_vector.reshape(reshape)
    lhs = float(np.sum(weighted))
    rhs = float(delta_score)
    error = abs(lhs - rhs)
    tolerance = 1e-3 * max(1.0, abs(rhs))

    return CheckResult(
        name="ig_completeness_error",
        value=error,
        passed=error <= tolerance,
    )


def check_smoothing_energy(
    original_map: TensorLike,
    smoothed_map: TensorLike,
    operators: MeshOperators,
) -> CheckResult:
    """Compare roughness energy before and after smoothing."""
    ops = coerce_operators(operators)
    mass = require_mass(ops)
    stiffness, laplacian = require_stiffness_or_laplacian(ops)

    n_vertices = infer_num_vertices_from_mass(mass)
    original = _to_vertex_first_matrix(original_map, n_vertices)
    smoothed = _to_vertex_first_matrix(smoothed_map, n_vertices)

    if original.shape != smoothed.shape:
        raise ValueError(
            "Original and smoothed maps must have the same shape. "
            f"Got {original.shape} vs {smoothed.shape}."
        )

    if stiffness is not None:
        energy_before = _quadratic_energy(stiffness, original)
        energy_after = _quadratic_energy(stiffness, smoothed)
    else:
        if laplacian is None:  # pragma: no cover - unreachable after require_* call
            raise ValueError("Missing Laplacian for smoothing energy check.")
        mass_vector = np.asarray(_to_numpy(mass_to_vector(mass)), dtype=np.float64).reshape(-1)
        weighted_laplacian = _left_multiply_diag(mass_vector, laplacian)
        energy_before = _quadratic_energy(weighted_laplacian, original)
        energy_after = _quadratic_energy(weighted_laplacian, smoothed)

    ratio = energy_after / max(energy_before, np.finfo(np.float64).eps)
    return CheckResult(
        name="smoothing_energy_ratio",
        value=float(ratio),
        passed=energy_after <= (energy_before + 1e-8),
    )


def _to_numpy(value: TensorLike) -> np.ndarray:
    if is_torch_tensor(value):
        return cast(np.ndarray, value.detach().cpu().numpy())
    return np.asarray(value)


def _matmul_operator(operator: TensorLike, values: np.ndarray) -> np.ndarray:
    if is_scipy_sparse(operator):
        sparse_op = cast(Any, operator)
        return np.asarray(sparse_op @ values, dtype=np.float64)
    matrix = np.asarray(_to_numpy(operator), dtype=np.float64)
    return np.asarray(matrix @ values, dtype=np.float64)


def _quadratic_energy(operator: TensorLike, values: np.ndarray) -> float:
    transformed = _matmul_operator(operator, values)
    return float(np.sum(values * transformed))


def _left_multiply_diag(diagonal: np.ndarray, operator: TensorLike) -> TensorLike:
    if is_scipy_sparse(operator):
        if sp is None:  # pragma: no cover
            raise RuntimeError("SciPy sparse backend unavailable.")
        return cast(TensorLike, sp.diags(diagonal) @ cast(Any, operator))

    matrix = np.asarray(_to_numpy(operator), dtype=np.float64)
    return diagonal.reshape(-1, 1) * matrix


def _to_vertex_first_matrix(value: TensorLike, n_vertices: int) -> np.ndarray:
    array = np.asarray(_to_numpy(value), dtype=np.float64)
    vertex_axis = infer_vertex_axis(array.shape, n_vertices)
    moved = np.moveaxis(array, vertex_axis, 0)
    return moved.reshape(n_vertices, -1)
