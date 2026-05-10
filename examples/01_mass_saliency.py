"""Example for mass-normalized saliency."""

from __future__ import annotations

import torch

from geoinspect import MeshOperators, SaliencyConfig, SaliencyExplainer


class ScalarLinearModel(torch.nn.Module):
    """Simple linear scalar model for demonstration."""

    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * self.weights)


if __name__ == "__main__":
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    mass = torch.tensor([1.5, 2.5], dtype=torch.float32)
    weights = torch.tensor([[0.2, -0.5], [1.0, 0.7]], dtype=torch.float32)

    explainer = SaliencyExplainer(
        model=ScalarLinearModel(weights),
        config=SaliencyConfig(aggregation="l2", mass_normalize=True),
    )

    result = explainer.explain(
        features=features,
        operators=MeshOperators(mass=mass),
        target=None,
    )

    print("raw_map shape:", tuple(result.raw_map.shape))
    print("density_map:", result.density_map)
    print("contribution_map:", result.contribution_map)
