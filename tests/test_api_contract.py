"""Public API contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from geoinspect import (
    BaselineConfig,
    GradCAMConfig,
    IntegratedGradientsConfig,
    IntegratedGradientsExplainer,
    IntrinsicGradCAMExplainer,
    MeshOperators,
    SaliencyConfig,
    SaliencyExplainer,
    build_baseline,
    normalize_map,
    smooth_map,
)
from geoinspect.operators import coerce_operators, mass_to_vector

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@pytest.mark.skipif(torch is None, reason="torch is required")
def test_gradcam_mass_weighted_signed_map_consistency() -> None:
    assert torch is not None

    class ToyGradCAMModel(torch.nn.Module):
        def __init__(self, weights: torch.Tensor) -> None:
            super().__init__()
            self.tap = torch.nn.Identity()
            self.register_buffer("weights", weights)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            activation = self.tap(x)
            return torch.sum(activation * self.weights)

    weights = torch.tensor([[2.0, 0.0], [0.0, 4.0]], dtype=torch.float32)
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    mass = torch.tensor([1.0, 3.0], dtype=torch.float32)

    explainer = IntrinsicGradCAMExplainer(
        model=ToyGradCAMModel(weights),
        config=GradCAMConfig(
            target_layer="tap",
            mass_weighted=True,
            use_relu=False,
            signed=True,
            return_channel_weights=True,
        ),
    )

    result = explainer.explain(
        features=features,
        operators=MeshOperators(mass=mass),
        target=None,
    )

    expected_alpha = torch.tensor([0.5, 3.0], dtype=torch.float32)
    expected_signed = torch.tensor([6.5, 13.5], dtype=torch.float32)
    expected_contrib = torch.tensor([6.5, 40.5], dtype=torch.float32)

    assert isinstance(result.raw_map, torch.Tensor)
    assert isinstance(result.density_map, torch.Tensor)
    assert isinstance(result.contribution_map, torch.Tensor)
    torch.testing.assert_close(result.raw_map, expected_signed)
    torch.testing.assert_close(result.density_map, expected_signed)
    torch.testing.assert_close(result.contribution_map, expected_contrib)

    alpha_meta = result.metadata.get("channel_weights")
    assert isinstance(alpha_meta, torch.Tensor)
    torch.testing.assert_close(alpha_meta, expected_alpha)


@pytest.mark.skipif(torch is None, reason="torch is required")
def test_gradcam_non_mass_weighted_relu_behavior() -> None:
    assert torch is not None

    class ToyGradCAMModel(torch.nn.Module):
        def __init__(self, weights: torch.Tensor) -> None:
            super().__init__()
            self.tap = torch.nn.Identity()
            self.register_buffer("weights", weights)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            activation = self.tap(x)
            return torch.sum(activation * self.weights)

    weights = torch.tensor([[2.0, 0.0], [0.0, 4.0]], dtype=torch.float32)
    features = torch.tensor([[-10.0, -1.0], [3.0, 4.0]], dtype=torch.float32)
    mass = torch.tensor([1.0, 3.0], dtype=torch.float32)

    explainer = IntrinsicGradCAMExplainer(
        model=ToyGradCAMModel(weights),
        config=GradCAMConfig(
            target_layer="tap",
            mass_weighted=False,
            use_relu=True,
            signed=False,
            return_channel_weights=False,
        ),
    )

    result = explainer.explain(
        features=features,
        operators=MeshOperators(mass=mass),
        target=None,
    )

    expected_signed = torch.tensor([-12.0, 11.0], dtype=torch.float32)
    expected_density = torch.tensor([0.0, 11.0], dtype=torch.float32)
    expected_contrib = torch.tensor([0.0, 33.0], dtype=torch.float32)

    assert isinstance(result.raw_map, torch.Tensor)
    assert isinstance(result.density_map, torch.Tensor)
    assert isinstance(result.contribution_map, torch.Tensor)
    torch.testing.assert_close(result.raw_map, expected_signed)
    torch.testing.assert_close(result.density_map, expected_density)
    torch.testing.assert_close(result.contribution_map, expected_contrib)


def test_coerce_operators_supports_diffusionnet_keys() -> None:
    ops = coerce_operators(
        {
            "mass": np.array([1.0, 2.0]),
            "L": np.eye(2),
            "evals": np.array([0.0, 1.0]),
            "evecs": np.eye(2),
            "gradX": np.eye(2),
            "gradY": np.eye(2),
        }
    )

    assert ops.mass is not None
    assert ops.laplacian is not None
    assert ops.grad_x is not None
    assert ops.grad_y is not None


def test_mass_to_vector_accepts_matrix_mass() -> None:
    mass_matrix = np.diag([2.0, 3.0, 4.0])
    mass_vec = mass_to_vector(mass_matrix)
    np.testing.assert_allclose(mass_vec, np.array([2.0, 3.0, 4.0]))


def test_normalize_map_minmax() -> None:
    ops = MeshOperators(mass=np.array([1.0, 1.0, 1.0]))
    result = normalize_map(np.array([-2.0, 0.0, 2.0]), operators=ops, method="minmax")
    np.testing.assert_allclose(result, np.array([0.0, 0.5, 1.0]))


def test_normalize_map_max_abs() -> None:
    ops = MeshOperators(mass=np.array([1.0, 1.0]))
    result = normalize_map(np.array([-2.0, 1.0]), operators=ops, method="max_abs")
    np.testing.assert_allclose(result, np.array([-1.0, 0.5]))


def test_normalize_map_area_integral() -> None:
    ops = MeshOperators(mass=np.array([2.0, 1.0]))
    result = normalize_map(np.array([1.0, 1.0]), operators=ops, method="area_integral")
    np.testing.assert_allclose(result, np.array([1.0 / 3.0, 1.0 / 3.0]))


def test_smooth_map_none_returns_copy() -> None:
    signal = np.array([1.0, 2.0, 3.0])
    ops = MeshOperators(mass=np.ones(3))
    smoothed = smooth_map(signal, operators=ops, method="none")

    np.testing.assert_allclose(smoothed, signal)
    assert smoothed is not signal


def test_smooth_map_heat_identity_basis() -> None:
    signal = np.array([1.0, 2.0])
    ops = MeshOperators(
        mass=np.ones(2),
        evals=np.array([0.0, 1.0]),
        evecs=np.eye(2),
    )
    smoothed = smooth_map(signal, operators=ops, method="heat", tau=1.0)
    expected = np.array([1.0, 2.0 * np.exp(-1.0)])
    np.testing.assert_allclose(smoothed, expected)


def test_smooth_map_helmholtz_zero_stiffness_returns_input() -> None:
    signal = np.array([3.0, -1.0])
    ops = MeshOperators(
        mass=np.ones(2),
        stiffness=np.zeros((2, 2)),
    )
    smoothed = smooth_map(signal, operators=ops, method="helmholtz", tau=0.5)
    np.testing.assert_allclose(smoothed, signal)


def test_build_baseline_zero_numpy() -> None:
    features = np.array([[1.0, 2.0], [3.0, 4.0]])
    baseline = build_baseline(features, MeshOperators(mass=np.ones(2)), BaselineConfig(kind="zero"))
    np.testing.assert_allclose(baseline, np.zeros_like(features))


def test_build_baseline_mean_numpy_mass_weighted() -> None:
    features = np.array([[2.0, 0.0], [4.0, 2.0]])
    baseline = build_baseline(
        features,
        MeshOperators(mass=np.array([1.0, 3.0])),
        BaselineConfig(kind="mean"),
    )
    expected_row = np.array([3.5, 1.5])
    expected = np.vstack([expected_row, expected_row])
    np.testing.assert_allclose(baseline, expected)


def test_build_baseline_spectral_lowpass_numpy_identity() -> None:
    features = np.array([[5.0], [7.0]])
    ops = MeshOperators(
        mass=np.ones(2),
        evals=np.array([0.0, 1.0]),
        evecs=np.eye(2),
    )
    baseline = build_baseline(
        features,
        ops,
        BaselineConfig(kind="spectral_lowpass", kwargs={"num_modes": 1}),
    )
    np.testing.assert_allclose(baseline, np.array([[5.0], [0.0]]))


@pytest.mark.skipif(torch is None, reason="torch is required")
def test_saliency_mass_normalized_l2_consistency() -> None:
    assert torch is not None

    class ScalarLinearModel(torch.nn.Module):
        def __init__(self, weights: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("weights", weights)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sum(x * self.weights)

    weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    features = torch.tensor([[5.0, -1.0], [2.0, 7.0]], dtype=torch.float32)
    mass = torch.tensor([2.0, 4.0], dtype=torch.float32)

    explainer = SaliencyExplainer(
        model=ScalarLinearModel(weights),
        config=SaliencyConfig(aggregation="l2", mass_normalize=True),
    )

    result = explainer.explain(features=features, operators=MeshOperators(mass=mass), target=None)

    expected_raw = weights
    expected_density = torch.tensor(
        [
            ((1.0 / 2.0) ** 2 + (2.0 / 2.0) ** 2) ** 0.5,
            ((3.0 / 4.0) ** 2 + (4.0 / 4.0) ** 2) ** 0.5,
        ],
        dtype=torch.float32,
    )
    expected_contrib = mass * expected_density

    assert isinstance(result.raw_map, torch.Tensor)
    assert isinstance(result.density_map, torch.Tensor)
    assert isinstance(result.contribution_map, torch.Tensor)

    torch.testing.assert_close(result.raw_map, expected_raw)
    torch.testing.assert_close(result.density_map, expected_density)
    torch.testing.assert_close(result.contribution_map, expected_contrib)


@pytest.mark.skipif(torch is None, reason="torch is required")
def test_saliency_signed_mean_baseline_consistency() -> None:
    assert torch is not None

    class ScalarLinearModel(torch.nn.Module):
        def __init__(self, weights: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("weights", weights)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sum(x * self.weights)

    weights = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    features = torch.tensor([[2.0, 0.0], [4.0, 2.0]], dtype=torch.float32)
    mass = torch.tensor([1.0, 3.0], dtype=torch.float32)

    explainer = SaliencyExplainer(
        model=ScalarLinearModel(weights),
        config=SaliencyConfig(aggregation="signed", mass_normalize=True, baseline="mean"),
    )

    result = explainer.explain(features=features, operators=MeshOperators(mass=mass), target=None)

    expected_density = torch.tensor([-3.0, 1.0 / 3.0], dtype=torch.float32)
    expected_contrib = torch.tensor([-3.0, 1.0], dtype=torch.float32)

    assert isinstance(result.density_map, torch.Tensor)
    assert isinstance(result.contribution_map, torch.Tensor)
    torch.testing.assert_close(result.density_map, expected_density)
    torch.testing.assert_close(result.contribution_map, expected_contrib)


@pytest.mark.skipif(torch is None, reason="torch is required")
def test_saliency_target_index_last_dimension() -> None:
    assert torch is not None

    class ClassScoreModel(torch.nn.Module):
        def __init__(self, weights: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("weights", weights)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.einsum("vc,ck->vk", x, self.weights)

    weights = torch.tensor([[1.0, 10.0], [2.0, 20.0]], dtype=torch.float32)
    features = torch.tensor([[1.0, 3.0], [4.0, 5.0]], dtype=torch.float32)
    mass = torch.tensor([1.0, 1.0], dtype=torch.float32)

    explainer = SaliencyExplainer(
        model=ClassScoreModel(weights),
        config=SaliencyConfig(aggregation="l1", mass_normalize=False),
    )

    result = explainer.explain(features=features, operators=MeshOperators(mass=mass), target=1)

    expected_raw = torch.tensor([[10.0, 20.0], [10.0, 20.0]], dtype=torch.float32)
    expected_density = torch.tensor([30.0, 30.0], dtype=torch.float32)

    assert isinstance(result.raw_map, torch.Tensor)
    assert isinstance(result.density_map, torch.Tensor)
    torch.testing.assert_close(result.raw_map, expected_raw)
    torch.testing.assert_close(result.density_map, expected_density)


@pytest.mark.skipif(torch is None, reason="torch is required")
def test_integrated_gradients_zero_baseline_mass_consistency() -> None:
    assert torch is not None

    class ScalarLinearModel(torch.nn.Module):
        def __init__(self, weights: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("weights", weights)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sum(x * self.weights)

    weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    features = torch.tensor([[2.0, 1.0], [5.0, 2.0]], dtype=torch.float32)
    mass = torch.tensor([2.0, 4.0], dtype=torch.float32)

    explainer = IntegratedGradientsExplainer(
        model=ScalarLinearModel(weights),
        config=IntegratedGradientsConfig(
            steps=24,
            baseline="zero",
            mass_normalize_gradients=True,
            return_channelwise=False,
        ),
    )

    result = explainer.explain(features=features, operators=MeshOperators(mass=mass), target=None)

    expected_density = torch.tensor(
        [
            (1.0 / 2.0) * 2.0 + (2.0 / 2.0) * 1.0,
            (3.0 / 4.0) * 5.0 + (4.0 / 4.0) * 2.0,
        ],
        dtype=torch.float32,
    )
    expected_contrib = mass * expected_density
    expected_delta = torch.sum(weights * features)

    assert isinstance(result.density_map, torch.Tensor)
    assert isinstance(result.contribution_map, torch.Tensor)
    torch.testing.assert_close(result.density_map, expected_density, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(result.contribution_map, expected_contrib, atol=1e-4, rtol=1e-4)

    completeness_error = float(result.metadata["completeness_error"])
    completeness_rhs = float(result.metadata["completeness_rhs"])
    assert abs(completeness_rhs - float(expected_delta)) < 1e-5
    assert completeness_error < 5e-3


@pytest.mark.skipif(torch is None, reason="torch is required")
def test_integrated_gradients_return_channelwise_shape() -> None:
    assert torch is not None

    class ScalarLinearModel(torch.nn.Module):
        def __init__(self, weights: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("weights", weights)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sum(x * self.weights)

    weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    mass = torch.tensor([1.0, 2.0], dtype=torch.float32)

    explainer = IntegratedGradientsExplainer(
        model=ScalarLinearModel(weights),
        config=IntegratedGradientsConfig(
            steps=16,
            baseline="zero",
            mass_normalize_gradients=False,
            return_channelwise=True,
        ),
    )

    result = explainer.explain(features=features, operators=MeshOperators(mass=mass), target=None)

    expected_channelwise = weights * features

    assert isinstance(result.density_map, torch.Tensor)
    assert isinstance(result.contribution_map, torch.Tensor)
    assert tuple(result.density_map.shape) == tuple(features.shape)
    torch.testing.assert_close(result.density_map, expected_channelwise, atol=1e-4, rtol=1e-4)

    expected_contribution = torch.tensor(
        [
            [1.0 * 1.0, 1.0 * 4.0],
            [2.0 * 9.0, 2.0 * 16.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(result.contribution_map, expected_contribution, atol=1e-4, rtol=1e-4)
