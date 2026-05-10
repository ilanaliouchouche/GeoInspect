"""Example for Integrated Gradients with spectral low-pass baseline."""

from __future__ import annotations

import torch

from geoinspect import (
    IntegratedGradientsConfig,
    IntegratedGradientsExplainer,
    MeshOperators,
)


class ScalarLinearModel(torch.nn.Module):
    """Simple linear scalar model for demonstration."""

    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * self.weights)


if __name__ == "__main__":
    features = torch.tensor([[2.0, 5.0], [7.0, 11.0]], dtype=torch.float32)
    mass = torch.tensor([1.0, 1.0], dtype=torch.float32)
    evals = torch.tensor([0.0, 2.0], dtype=torch.float32)
    evecs = torch.eye(2, dtype=torch.float32)

    config = IntegratedGradientsConfig(
        steps=24,
        baseline="spectral_lowpass",
        baseline_kwargs={"num_modes": 1},
        mass_normalize_gradients=True,
    )
    explainer = IntegratedGradientsExplainer(
        model=ScalarLinearModel(torch.tensor([[1.0, 0.5], [0.5, 2.0]], dtype=torch.float32)),
        config=config,
    )

    operators = MeshOperators(mass=mass, evals=evals, evecs=evecs)
    result = explainer.explain(features=features, operators=operators, target=None)

    print("density_map:", result.density_map)
    print("completeness_error:", result.metadata["completeness_error"])
