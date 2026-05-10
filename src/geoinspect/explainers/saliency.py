"""Mass-normalized saliency explainer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from ..forward import ForwardConfig, run_model_forward
from ..normalization import normalize_map
from ..operators import (
    MeshOperators,
    coerce_operators,
    infer_num_vertices_from_mass,
    mass_to_vector,
    require_mass,
)
from ..results import ExplanationResult
from ..smoothing import smooth_map
from ..targets import TargetSpec, resolve_target
from ..types import TargetLike, TensorLike
from .base import BaseExplainer, ExplainerConfig

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class SaliencyConfig(ExplainerConfig):
    """Configuration for mass-normalized saliency."""

    aggregation: str = "l2"
    mass_normalize: bool = True
    baseline: str | None = None


class SaliencyExplainer(BaseExplainer):
    """Compute saliency maps adapted to mesh geometry."""

    def __init__(self, model: TensorLike, config: SaliencyConfig | None = None) -> None:
        super().__init__(model=model, config=config or SaliencyConfig())

    def explain(
        self,
        features: TensorLike,
        operators: MeshOperators,
        target: TargetLike,
    ) -> ExplanationResult:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for gradient-based saliency explainers.")
        if not isinstance(features, torch.Tensor):
            raise TypeError("`features` must be a torch.Tensor for saliency explainers.")

        config = cast(SaliencyConfig, self.config)
        aggregation = _canonical_aggregation(config.aggregation)
        ops = coerce_operators(operators)
        forward_config = ForwardConfig(
            kwargs=dict(config.forward_kwargs),
            prefer_operator_signature=config.prefer_operator_signature,
        )

        feature_input = features.detach().clone()
        if not feature_input.is_floating_point():
            feature_input = feature_input.to(dtype=torch.float32)
        feature_input.requires_grad_(True)

        model_output = _call_model(self.model, feature_input, ops, forward_config)
        score = resolve_target(model_output, TargetSpec(target=target))

        gradients = torch.autograd.grad(
            score,
            feature_input,
            retain_graph=False,
            create_graph=False,
        )[0]

        raw_map = gradients.detach().clone()

        density_tensor = _mass_normalize_if_requested(raw_map, ops, config.mass_normalize)
        density_scalar = _aggregate_density(
            density_tensor=density_tensor,
            features=feature_input.detach(),
            operators=ops,
            aggregation=aggregation,
            baseline_kind=config.baseline,
        )

        normalized_density = (
            normalize_map(density_scalar, ops, method=config.normalize)
            if config.normalize not in {None, "", "none"}
            else density_scalar.clone()
        )

        contribution_map = _build_contribution_map(normalized_density, ops)

        if config.smooth in {None, "", "none"}:
            smoothed_map = None
        else:
            smooth_method = cast(str, config.smooth)
            smoothed = smooth_map(
                normalized_density,
                ops,
                method=smooth_method,
                tau=config.smooth_tau,
                num_modes=config.smooth_num_modes,
            )
            smoothed_map = cast(torch.Tensor, smoothed)
            if config.normalize not in {None, "", "none"}:
                smoothed_map = cast(
                    torch.Tensor,
                    normalize_map(smoothed_map, ops, method=config.normalize),
                )

        metadata: dict[str, object] = {
            "method": "mass_normalized_saliency",
            "aggregation": aggregation,
            "mass_normalize": config.mass_normalize,
            "target": _target_to_metadata(target),
            "baseline": config.baseline,
            "smoothing": config.smooth,
            "smooth_tau": config.smooth_tau,
            "smooth_num_modes": config.smooth_num_modes,
            "normalize": config.normalize,
            "score": float(score.detach().cpu().item()),
            "input_shape": tuple(int(v) for v in feature_input.shape),
            "forward_kwargs_keys": sorted(config.forward_kwargs.keys()),
            "prefer_operator_signature": config.prefer_operator_signature,
        }

        return ExplanationResult(
            raw_map=raw_map,
            density_map=normalized_density,
            contribution_map=contribution_map,
            smoothed_map=smoothed_map,
            metadata=metadata,
        )


def _call_model(
    model: TensorLike,
    features: torch.Tensor,
    operators: MeshOperators,
    forward_config: ForwardConfig,
) -> torch.Tensor:
    if not callable(model):
        raise TypeError("`model` must be callable.")
    output = run_model_forward(
        model,
        features,
        operators=operators,
        forward_config=forward_config,
    )
    if not isinstance(output, torch.Tensor):
        raise TypeError("Model output must be a torch.Tensor.")
    return output


def _canonical_aggregation(aggregation: str) -> str:
    method = aggregation.strip().lower()
    if method not in {"l2", "l1", "abs_sum", "signed"}:
        raise ValueError("`aggregation` must be one of: l2, l1, abs_sum, signed.")
    return method


def _mass_normalize_if_requested(
    gradients: torch.Tensor,
    operators: MeshOperators,
    use_mass_normalization: bool,
) -> torch.Tensor:
    if not use_mass_normalization:
        return gradients.detach().clone()

    mass = require_mass(operators)
    n_vertices = infer_num_vertices_from_mass(mass)
    vertex_axis = _vertex_axis_from_saliency_shape(tuple(gradients.shape), n_vertices)
    mass_vector = mass_to_vector(mass)

    if isinstance(mass_vector, torch.Tensor):
        mass_tensor = mass_vector.to(dtype=gradients.dtype, device=gradients.device)
    else:
        mass_tensor = torch.as_tensor(mass_vector, dtype=gradients.dtype, device=gradients.device)

    if torch.any(mass_tensor <= 0):
        raise ValueError("Mass entries must be strictly positive for mass normalization.")

    reshape = [1] * gradients.ndim
    reshape[vertex_axis] = n_vertices
    return gradients / mass_tensor.reshape(reshape)


def _aggregate_density(
    density_tensor: torch.Tensor,
    features: torch.Tensor,
    operators: MeshOperators,
    aggregation: str,
    baseline_kind: str | None,
) -> torch.Tensor:
    n_vertices = infer_num_vertices_from_mass(require_mass(operators))
    vertex_axis = _vertex_axis_from_saliency_shape(tuple(density_tensor.shape), n_vertices)

    if aggregation == "signed":
        baseline = _build_baseline(features, operators, baseline_kind)
        delta = features - baseline
        return _sum_over_channels(density_tensor * delta, vertex_axis)

    abs_density = torch.abs(density_tensor)

    if aggregation == "l2":
        squared = density_tensor * density_tensor
        return torch.sqrt(_sum_over_channels(squared, vertex_axis))
    if aggregation == "l1":
        return _sum_over_channels(abs_density, vertex_axis)
    if aggregation == "abs_sum":
        return _sum_over_channels(abs_density, vertex_axis)

    raise AssertionError("Unreachable aggregation branch.")


def _sum_over_channels(value: torch.Tensor, vertex_axis: int) -> torch.Tensor:
    channel_axes = [axis for axis in range(value.ndim) if axis > vertex_axis]
    if not channel_axes:
        return value

    reduced = value
    for axis in sorted(channel_axes, reverse=True):
        reduced = torch.sum(reduced, dim=axis)
    return reduced


def _build_baseline(
    features: torch.Tensor,
    operators: MeshOperators,
    baseline_kind: str | None,
) -> torch.Tensor:
    kind = "zero" if baseline_kind is None else baseline_kind.strip().lower()

    if kind == "zero":
        return torch.zeros_like(features)

    if kind == "mean":
        mass = require_mass(operators)
        n_vertices = infer_num_vertices_from_mass(mass)
        vertex_axis = _vertex_axis_from_saliency_shape(tuple(features.shape), n_vertices)
        mass_vector = mass_to_vector(mass)

        if isinstance(mass_vector, torch.Tensor):
            mass_tensor = mass_vector.to(dtype=features.dtype, device=features.device)
        else:
            mass_tensor = torch.as_tensor(mass_vector, dtype=features.dtype, device=features.device)

        reshape = [1] * features.ndim
        reshape[vertex_axis] = n_vertices
        weighted = features * mass_tensor.reshape(reshape)
        denominator = torch.sum(mass_tensor)
        if denominator <= 0:
            raise ValueError("Mass vector must have strictly positive sum for mean baseline.")

        mean_feature = weighted.sum(dim=vertex_axis, keepdim=True) / denominator
        return torch.ones_like(features) * mean_feature

    if kind == "heat":
        smoothed = smooth_map(features.detach(), operators, method="heat")
        return cast(torch.Tensor, smoothed)

    if kind == "spectral_lowpass":
        smoothed = smooth_map(features.detach(), operators, method="heat", tau=0.0, num_modes=32)
        return cast(torch.Tensor, smoothed)

    raise ValueError("`baseline` must be one of: None, zero, mean, heat, spectral_lowpass.")


def _build_contribution_map(density_map: torch.Tensor, operators: MeshOperators) -> torch.Tensor:
    mass = require_mass(operators)
    n_vertices = infer_num_vertices_from_mass(mass)
    vertex_axis = _vertex_axis_from_saliency_shape(tuple(density_map.shape), n_vertices)
    mass_vector = mass_to_vector(mass)

    if isinstance(mass_vector, torch.Tensor):
        mass_tensor = mass_vector.to(dtype=density_map.dtype, device=density_map.device)
    else:
        mass_tensor = torch.as_tensor(
            mass_vector,
            dtype=density_map.dtype,
            device=density_map.device,
        )

    reshape = [1] * density_map.ndim
    reshape[vertex_axis] = n_vertices
    return density_map * mass_tensor.reshape(reshape)


def _target_to_metadata(target: object) -> object:
    if callable(target):
        fn = cast(Callable[[torch.Tensor], TensorLike], target)
        return f"callable:{getattr(fn, '__name__', 'anonymous')}"
    return target


def _vertex_axis_from_saliency_shape(shape: tuple[int, ...], n_vertices: int) -> int:
    if len(shape) == 1:
        if shape[0] != n_vertices:
            raise ValueError(f"Expected shape [n] with n={n_vertices}, got {shape}.")
        return 0

    if len(shape) == 2:
        if shape[0] == n_vertices and shape[1] != n_vertices:
            return 0
        if shape[1] == n_vertices and shape[0] != n_vertices:
            return 1
        if shape[0] == n_vertices and shape[1] == n_vertices:
            return 0
        raise ValueError(f"Expected shape [n, C] or [B, n], with n={n_vertices}, got {shape}.")

    if len(shape) == 3:
        if shape[1] == n_vertices:
            return 1
        if shape[0] == n_vertices:
            return 0
        if shape[2] == n_vertices:
            return 2
        raise ValueError(f"Expected shape [B, n, C] with n={n_vertices}, got {shape}.")

    raise ValueError(
        f"Unsupported saliency tensor rank {len(shape)} for shape {shape}. "
        "Expected [n,C] or [B,n,C]."
    )
