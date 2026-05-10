"""Interactive smoothing comparison with Polyscope and tau slider."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch

from geoinspect import (
    IntegratedGradientsConfig,
    IntegratedGradientsExplainer,
    MeshOperators,
    PolyscopeViewerConfig,
    launch_polyscope_viewer,
)

JSONLike: TypeAlias = (
    dict[str, "JSONLike"] | list["JSONLike"] | str | int | float | bool | None
)


class ToyDiffusionNetLike(torch.nn.Module):
    """Toy model exposing a DiffusionNet-style forward signature."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.head = torch.nn.Linear(in_channels, 1, bias=False)

    def forward(
        self,
        features: torch.Tensor,
        mass: torch.Tensor,
        laplacian: torch.Tensor,
        evals: torch.Tensor,
        evecs: torch.Tensor,
        grad_x: torch.Tensor,
        grad_y: torch.Tensor,
    ) -> torch.Tensor:
        del evals, evecs, grad_x, grad_y
        mass_vec = mass if mass.ndim == 1 else torch.diagonal(mass)
        diffused = features - 0.15 * (laplacian @ features)
        weighted = diffused * mass_vec.reshape(-1, 1)
        logits = self.head(weighted).reshape(-1)
        return torch.sum(logits)


def _to_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_jsonable(value: object) -> JSONLike:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)

    features = torch.tensor(
        [
            [0.1, 1.0, 0.2],
            [0.6, 0.2, 0.4],
            [0.9, 0.8, 0.1],
            [0.3, 0.5, 0.7],
        ],
        dtype=torch.float32,
    )

    mass = torch.ones(4, dtype=torch.float32)
    laplacian = torch.tensor(
        [
            [2.0, -1.0, 0.0, -1.0],
            [-1.0, 2.0, -1.0, 0.0],
            [0.0, -1.0, 2.0, -1.0],
            [-1.0, 0.0, -1.0, 2.0],
        ],
        dtype=torch.float32,
    )
    evals, evecs = torch.linalg.eigh(laplacian)
    grad_x = torch.zeros((4, 4), dtype=torch.float32)
    grad_y = torch.zeros((4, 4), dtype=torch.float32)

    operators = MeshOperators(
        mass=mass,
        laplacian=laplacian,
        evals=evals,
        evecs=evecs,
        grad_x=grad_x,
        grad_y=grad_y,
    )

    model = ToyDiffusionNetLike(in_channels=int(features.shape[1]))
    with torch.no_grad():
        model.head.weight.copy_(torch.tensor([[1.0, -0.3, 0.7]], dtype=torch.float32))

    explainer = IntegratedGradientsExplainer(
        model=model,
        config=IntegratedGradientsConfig(
            steps=48,
            baseline="heat",
            baseline_kwargs={"tau": 0.04},
            mass_normalize_gradients=True,
            prefer_operator_signature=True,
        ),
    )
    result = explainer.explain(features=features, operators=operators, target=None)

    output_dir = Path("outputs/example05")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "maps.npz",
        raw_map=_to_numpy(result.raw_map),
        density_map=_to_numpy(result.density_map),
        contribution_map=_to_numpy(result.contribution_map),
        smoothed_map=(
            np.array([], dtype=np.float32)
            if result.smoothed_map is None
            else _to_numpy(result.smoothed_map)
        ),
    )
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(result.metadata), f, indent=2)

    print("Saved artifacts in:", output_dir)
    print("completeness_error:", result.metadata["completeness_error"])
    print("Open Polyscope: interactive slider t (tau) controls smoothing strength.")

    viewer_config = PolyscopeViewerConfig(
        mesh_name="example05_mesh",
        quantity_name="ig_density",
        smoothing_method="heat",
        initial_tau=0.02,
        tau_min=0.0,
        tau_max=0.5,
        num_modes=4,
        collapse="l2",
    )
    launch_polyscope_viewer(
        vertices=vertices,
        faces=faces,
        scalar_map=result.density_map,
        operators=operators,
        config=viewer_config,
    )
