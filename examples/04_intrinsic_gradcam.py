"""Example for intrinsic Grad-CAM on mesh-like activations."""

from __future__ import annotations

import torch

from geoinspect import GradCAMConfig, IntrinsicGradCAMExplainer, MeshOperators


class ToyGradCAMModel(torch.nn.Module):
    """Tiny model exposing an intermediate layer for Grad-CAM."""

    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.tap = torch.nn.Identity()
        self.register_buffer("weights", weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation = self.tap(x)
        return torch.sum(activation * self.weights)


if __name__ == "__main__":
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    mass = torch.tensor([1.0, 3.0], dtype=torch.float32)

    model = ToyGradCAMModel(weights=torch.tensor([[2.0, 0.0], [0.0, 4.0]], dtype=torch.float32))
    config = GradCAMConfig(target_layer="tap", mass_weighted=True, signed=True, use_relu=False)

    explainer = IntrinsicGradCAMExplainer(model=model, config=config)
    result = explainer.explain(features=features, operators=MeshOperators(mass=mass), target=None)

    print("raw_map:", result.raw_map)
    print("density_map:", result.density_map)
    print("contribution_map:", result.contribution_map)
    print("channel_weights:", result.metadata.get("channel_weights"))
