"""Tests for target resolution, forward adapters, and scientific checks."""

from __future__ import annotations

import numpy as np
import pytest

from geoinspect import (
    ForwardConfig,
    MeshOperators,
    TargetSpec,
    check_constant_signal,
    check_ig_completeness,
    check_mass_consistency,
    check_smoothing_energy,
    prepare_vertex_scalar_map,
    resolve_target,
    run_model_forward,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@pytest.mark.skipif(torch is None, reason="torch is required")
def test_resolve_target_none_scalar_output() -> None:
    assert torch is not None
    output = torch.tensor(2.5, dtype=torch.float32)
    score = resolve_target(output, TargetSpec(target=None))
    assert isinstance(score, torch.Tensor)
    assert score.ndim == 0
    assert float(score.item()) == pytest.approx(2.5)


@pytest.mark.skipif(torch is None, reason="torch is required")
def test_resolve_target_integer_index() -> None:
    assert torch is not None
    output = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    score = resolve_target(output, TargetSpec(target=1))
    assert isinstance(score, torch.Tensor)
    assert score.ndim == 0
    assert float(score.item()) == pytest.approx(6.0)


@pytest.mark.skipif(torch is None, reason="torch is required")
def test_resolve_target_callable() -> None:
    assert torch is not None
    output = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    score = resolve_target(output, TargetSpec(target_fn=lambda y: y[0, 1] - y[1, 0]))
    assert isinstance(score, torch.Tensor)
    assert float(score.item()) == pytest.approx(-1.0)


@pytest.mark.skipif(torch is None, reason="torch is required")
def test_run_model_forward_diffusionnet_signature() -> None:
    assert torch is not None

    class DiffusionStyleModel(torch.nn.Module):
        def forward(
            self,
            x: torch.Tensor,
            mass: torch.Tensor,
            laplacian: torch.Tensor,
            evals: torch.Tensor,
            evecs: torch.Tensor,
            grad_x: torch.Tensor,
            grad_y: torch.Tensor,
        ) -> torch.Tensor:
            del evals, evecs, grad_x, grad_y
            return torch.sum(x + mass.reshape(-1, 1) + 0.1 * (laplacian @ x))

    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    operators = MeshOperators(
        mass=torch.tensor([1.0, 2.0], dtype=torch.float32),
        laplacian=torch.eye(2, dtype=torch.float32),
        evals=torch.tensor([0.0, 1.0], dtype=torch.float32),
        evecs=torch.eye(2, dtype=torch.float32),
        grad_x=torch.zeros((2, 2), dtype=torch.float32),
        grad_y=torch.zeros((2, 2), dtype=torch.float32),
    )

    output = run_model_forward(
        DiffusionStyleModel(),
        features,
        operators=operators,
        forward_config=ForwardConfig(prefer_operator_signature=True),
    )
    assert isinstance(output, torch.Tensor)
    assert output.ndim == 0


@pytest.mark.skipif(torch is None, reason="torch is required")
def test_run_model_forward_plain_with_kwargs() -> None:
    assert torch is not None

    class ScaleModel(torch.nn.Module):
        def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
            return torch.sum(scale * x)

    features = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    output = run_model_forward(
        ScaleModel(),
        features,
        forward_config=ForwardConfig(kwargs={"scale": 2.0}),
    )
    assert isinstance(output, torch.Tensor)
    assert float(output.item()) == pytest.approx(6.0)


def test_check_mass_consistency_positive() -> None:
    result = check_mass_consistency(MeshOperators(mass=np.array([1.0, 2.0, 3.0])))
    assert result.name == "mass_consistency_total_area"
    assert result.passed
    assert result.value == pytest.approx(6.0)


def test_check_constant_signal_zero_stiffness() -> None:
    result = check_constant_signal(
        MeshOperators(
            mass=np.ones(3),
            stiffness=np.zeros((3, 3)),
        )
    )
    assert result.name == "constant_signal_response"
    assert result.passed
    assert result.value == pytest.approx(0.0)


def test_check_ig_completeness_exact() -> None:
    density = np.array([1.0, 1.0])
    operators = MeshOperators(mass=np.array([1.0, 2.0]))
    result = check_ig_completeness(density, operators, delta_score=3.0)
    assert result.name == "ig_completeness_error"
    assert result.passed
    assert result.value == pytest.approx(0.0)


def test_check_smoothing_energy_decreases() -> None:
    operators = MeshOperators(
        mass=np.ones(2),
        stiffness=np.array([[1.0, -1.0], [-1.0, 1.0]]),
    )
    original = np.array([1.0, -1.0])
    smoothed = np.array([0.0, 0.0])

    result = check_smoothing_energy(original, smoothed, operators)
    assert result.name == "smoothing_energy_ratio"
    assert result.passed
    assert result.value == pytest.approx(0.0)


def test_prepare_vertex_scalar_map_l2() -> None:
    values = np.array([[3.0, 4.0, 0.0], [5.0, 12.0, 0.0]])
    reduced = prepare_vertex_scalar_map(values, n_vertices=2, collapse="l2")
    np.testing.assert_allclose(reduced, np.array([5.0, 13.0]))
