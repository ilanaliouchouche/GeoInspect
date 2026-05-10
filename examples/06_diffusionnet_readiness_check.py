"""Run a reproducible DiffusionNet-readiness certification for GeoInspect."""

from __future__ import annotations

import numpy as np
import torch

from geoinspect import (
    GradCAMConfig,
    IntegratedGradientsConfig,
    IntegratedGradientsExplainer,
    IntrinsicGradCAMExplainer,
    MeshOperators,
    PolyscopeViewerConfig,
    SaliencyConfig,
    SaliencyExplainer,
    check_constant_signal,
    check_ig_completeness,
    check_mass_consistency,
    check_smoothing_energy,
    launch_polyscope_viewer,
)


class TinyDiffusionNet(torch.nn.Module):
    """Small DiffusionNet-style model with an intermediate tap layer."""

    def __init__(self, c_in: int, c_hid: int, c_out: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(c_in, c_hid)
        self.tap = torch.nn.Identity()
        self.head = torch.nn.Linear(c_hid, c_out)

    def forward(
        self,
        features: torch.Tensor,
        mass: torch.Tensor,
        laplacian: torch.Tensor,
        evals: torch.Tensor,
        evecs: torch.Tensor,
        grad_x: torch.Tensor,
        grad_y: torch.Tensor,
        faces: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del evals, evecs, grad_x, grad_y, faces
        mass_vec = mass if mass.ndim == 1 else torch.diagonal(mass)
        x = self.proj(features)
        x = x - 0.1 * (laplacian @ x)
        x = x * mass_vec.reshape(-1, 1)
        x = self.tap(x)
        logits = self.head(x)
        return logits.mean(dim=0)


def _target_class_1(out: torch.Tensor) -> torch.Tensor:
    return out[1]


def _build_demo_operators() -> tuple[MeshOperators, torch.Tensor, torch.Tensor]:
    n = 5
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.2, 0.9, 0.0],
            [0.2, 1.1, 0.0],
            [0.5, 0.4, 0.3],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 1, 2], [0, 2, 3]],
        dtype=torch.int64,
    )

    mass = torch.ones(n, dtype=torch.float32)
    laplacian = torch.tensor(
        [
            [3.0, -1.0, 0.0, -1.0, -1.0],
            [-1.0, 3.0, -1.0, 0.0, -1.0],
            [0.0, -1.0, 3.0, -1.0, -1.0],
            [-1.0, 0.0, -1.0, 3.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0, 4.0],
        ],
        dtype=torch.float32,
    )
    evals, evecs = torch.linalg.eigh(laplacian)
    grad_x = torch.zeros((n, n), dtype=torch.float32)
    grad_y = torch.zeros((n, n), dtype=torch.float32)

    operators = MeshOperators(
        mass=mass,
        laplacian=laplacian,
        stiffness=laplacian,
        evals=evals,
        evecs=evecs,
        grad_x=grad_x,
        grad_y=grad_y,
    )
    return operators, vertices, faces


def main() -> int:
    torch.manual_seed(0)

    operators, vertices, faces = _build_demo_operators()
    features = torch.randn((5, 4), dtype=torch.float32)
    model = TinyDiffusionNet(c_in=4, c_hid=8, c_out=3)

    saliency = SaliencyExplainer(
        model,
        SaliencyConfig(
            aggregation="l2",
            mass_normalize=True,
            smooth="heat",
            smooth_tau=0.03,
            smooth_num_modes=5,
            prefer_operator_signature=True,
            forward_kwargs={"faces": faces},
        ),
    )
    saliency_result = saliency.explain(features, operators, target=_target_class_1)

    integrated_gradients = IntegratedGradientsExplainer(
        model,
        IntegratedGradientsConfig(
            steps=24,
            baseline="heat",
            baseline_kwargs={"tau": 0.03},
            mass_normalize_gradients=True,
            smooth="helmholtz",
            smooth_tau=0.02,
            prefer_operator_signature=True,
            forward_kwargs={"faces": faces},
        ),
    )
    ig_result = integrated_gradients.explain(features, operators, target=_target_class_1)

    gradcam = IntrinsicGradCAMExplainer(
        model,
        GradCAMConfig(
            target_layer="tap",
            mass_weighted=True,
            signed=True,
            use_relu=False,
            smooth="heat",
            smooth_tau=0.02,
            prefer_operator_signature=True,
            forward_kwargs={"faces": faces},
        ),
    )
    gradcam_result = gradcam.explain(features, operators, target=_target_class_1)

    mass_check = check_mass_consistency(operators)
    constant_check = check_constant_signal(operators)
    ig_check = check_ig_completeness(
        ig_result.density_map,
        operators,
        float(ig_result.metadata["completeness_rhs"]),
    )
    smooth_map_for_check = (
        saliency_result.smoothed_map
        if saliency_result.smoothed_map is not None
        else saliency_result.density_map
    )
    smoothing_check = check_smoothing_energy(
        np.asarray(saliency_result.density_map.detach().cpu().numpy()),
        np.asarray(smooth_map_for_check.detach().cpu().numpy()),
        operators,
    )

    launch_polyscope_viewer(
        vertices=vertices,
        faces=faces,
        scalar_map=ig_result.density_map,
        operators=operators,
        config=PolyscopeViewerConfig(
            show=False,
            smoothing_method="heat",
            initial_tau=0.01,
            tau_max=0.1,
        ),
    )

    ok = all(
        [
            mass_check.passed,
            constant_check.passed,
            ig_check.passed,
            smoothing_check.passed,
            "channel_weights" in gradcam_result.metadata,
        ]
    )

    print("== DiffusionNet readiness ==")
    print("Saliency density shape:", tuple(saliency_result.density_map.shape))
    print("IG density shape:", tuple(ig_result.density_map.shape))
    print("IG completeness error:", float(ig_result.metadata["completeness_error"]))
    print("Grad-CAM density shape:", tuple(gradcam_result.density_map.shape))
    print(
        "Checks passed:",
        mass_check.passed,
        constant_check.passed,
        ig_check.passed,
        smoothing_check.passed,
    )
    print("STATUS:", "PASS" if ok else "FAIL")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
