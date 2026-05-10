"""Surface Integrated Gradients explainer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

from ..baselines import BaselineConfig, build_baseline
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
from ..types import MetadataDict, TargetLike, TensorLike
from .base import BaseExplainer, ExplainerConfig

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

try:
    from captum.attr import (  # type: ignore[import-not-found]
        IntegratedGradients as CaptumIntegratedGradients,
    )
except ImportError:  # pragma: no cover - optional dependency
    CaptumIntegratedGradients = None


@dataclass(slots=True)
class IntegratedGradientsConfig(ExplainerConfig):
    """Configuration for surface Integrated Gradients."""

    steps: int = 32
    baseline: str = "zero"
    baseline_kwargs: MetadataDict = field(default_factory=dict)
    mass_normalize_gradients: bool = True
    return_channelwise: bool = False


class IntegratedGradientsExplainer(BaseExplainer):
    """Compute mesh-aware Integrated Gradients explanations."""

    def __init__(
        self,
        model: TensorLike,
        config: IntegratedGradientsConfig | None = None,
    ) -> None:
        super().__init__(model=model, config=config or IntegratedGradientsConfig())

    def explain(
        self,
        features: TensorLike,
        operators: MeshOperators,
        target: TargetLike,
    ) -> ExplanationResult:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for Integrated Gradients.")
        if not isinstance(features, torch.Tensor):
            raise TypeError("`features` must be a torch.Tensor for Integrated Gradients.")

        config = cast(IntegratedGradientsConfig, self.config)
        if config.steps <= 0:
            raise ValueError("`steps` must be a strictly positive integer.")

        ops = coerce_operators(operators)
        forward_config = ForwardConfig(
            kwargs=dict(config.forward_kwargs),
            prefer_operator_signature=config.prefer_operator_signature,
        )

        feature_input = features.detach().clone()
        if not feature_input.is_floating_point():
            feature_input = feature_input.to(dtype=torch.float32)

        baseline_cfg = BaselineConfig(kind=config.baseline, kwargs=dict(config.baseline_kwargs))
        baseline_value = build_baseline(feature_input.detach(), ops, baseline_cfg)

        if isinstance(baseline_value, torch.Tensor):
            baseline = baseline_value.to(dtype=feature_input.dtype, device=feature_input.device)
        else:
            baseline = torch.as_tensor(
                baseline_value,
                dtype=feature_input.dtype,
                device=feature_input.device,
            )

        if baseline.shape != feature_input.shape:
            raise ValueError(
                "Baseline shape must match input shape. "
                f"Got baseline {tuple(baseline.shape)} vs input {tuple(feature_input.shape)}."
            )

        raw_avg_grad = _compute_average_gradients(
            model=self.model,
            features=feature_input,
            baseline=baseline,
            operators=ops,
            forward_config=forward_config,
            target=target,
            steps=config.steps,
        )

        density_grad = (
            _mass_normalize_gradient(raw_avg_grad, ops)
            if config.mass_normalize_gradients
            else raw_avg_grad.detach().clone()
        )

        delta = feature_input - baseline
        density_channelwise = density_grad * delta

        n_vertices = infer_num_vertices_from_mass(require_mass(ops))
        vertex_axis = _vertex_axis_from_feature_shape(tuple(density_channelwise.shape), n_vertices)
        density_scalar = _sum_over_channels(density_channelwise, vertex_axis)

        completeness_lhs = _mass_weighted_total(density_scalar, ops)
        score_input = resolve_target(
            _call_model(
                self.model,
                feature_input.detach(),
                operators=ops,
                forward_config=forward_config,
            ),
            TargetSpec(target=target),
        )
        score_baseline = resolve_target(
            _call_model(
                self.model,
                baseline.detach(),
                operators=ops,
                forward_config=forward_config,
            ),
            TargetSpec(target=target),
        )
        completeness_rhs = score_input - score_baseline
        completeness_error = torch.abs(completeness_lhs - completeness_rhs)

        if config.return_channelwise:
            density_out = density_channelwise.clone()
        else:
            density_out = density_scalar.clone()

        if config.normalize not in {None, "", "none"}:
            density_out = cast(
                torch.Tensor,
                normalize_map(density_out, ops, method=config.normalize),
            )

        contribution_out = _multiply_mass(density_out, ops)

        if config.smooth in {None, "", "none"}:
            smoothed_map = None
        else:
            smooth_method = cast(str, config.smooth)
            smoothed_map = cast(
                torch.Tensor,
                smooth_map(
                    density_out,
                    ops,
                    method=smooth_method,
                    tau=config.smooth_tau,
                    num_modes=config.smooth_num_modes,
                ),
            )
            if config.normalize not in {None, "", "none"}:
                smoothed_map = cast(
                    torch.Tensor,
                    normalize_map(smoothed_map, ops, method=config.normalize),
                )

        metadata: dict[str, object] = {
            "method": "surface_integrated_gradients",
            "steps": config.steps,
            "baseline": config.baseline,
            "baseline_kwargs": dict(config.baseline_kwargs),
            "mass_normalize_gradients": config.mass_normalize_gradients,
            "return_channelwise": config.return_channelwise,
            "target": _target_to_metadata(target),
            "smoothing": config.smooth,
            "smooth_tau": config.smooth_tau,
            "smooth_num_modes": config.smooth_num_modes,
            "normalize": config.normalize,
            "completeness_lhs": float(completeness_lhs.detach().cpu().item()),
            "completeness_rhs": float(completeness_rhs.detach().cpu().item()),
            "completeness_error": float(completeness_error.detach().cpu().item()),
            "forward_kwargs_keys": sorted(config.forward_kwargs.keys()),
            "prefer_operator_signature": config.prefer_operator_signature,
        }

        return ExplanationResult(
            raw_map=raw_avg_grad.detach().clone(),
            density_map=density_out,
            contribution_map=contribution_out,
            smoothed_map=smoothed_map,
            metadata=metadata,
        )


def _compute_average_gradients(
    model: TensorLike,
    features: torch.Tensor,
    baseline: torch.Tensor,
    operators: MeshOperators,
    forward_config: ForwardConfig,
    target: TargetLike,
    steps: int,
) -> torch.Tensor:
    if CaptumIntegratedGradients is not None:
        forward_fn = _make_target_forward(model, operators, forward_config, target)
        captum_ig = CaptumIntegratedGradients(forward_fn, multiply_by_inputs=False)
        integrated_grads = captum_ig.attribute(
            features,
            baselines=baseline,
            n_steps=steps,
            method="gausslegendre",
        )
        if not isinstance(integrated_grads, torch.Tensor):
            raise TypeError("Captum IntegratedGradients returned unsupported attribution type.")
        return integrated_grads.detach().clone()

    delta = features - baseline
    grad_accumulator = torch.zeros_like(features)
    for step_idx in range(1, steps + 1):
        alpha = float(step_idx) / float(steps)
        point = baseline + alpha * delta
        point = point.detach().clone().requires_grad_(True)
        score = resolve_target(
            _call_model(model, point, operators=operators, forward_config=forward_config),
            TargetSpec(target=target),
        )
        grads = torch.autograd.grad(score, point, retain_graph=False, create_graph=False)[0]
        grad_accumulator = grad_accumulator + grads.detach()
    return grad_accumulator / float(steps)


def _make_target_forward(
    model: TensorLike,
    operators: MeshOperators,
    forward_config: ForwardConfig,
    target: TargetLike,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def _forward(inputs: torch.Tensor) -> torch.Tensor:
        output = _call_model(model, inputs, operators=operators, forward_config=forward_config)
        score = resolve_target(output, TargetSpec(target=target))
        if not isinstance(score, torch.Tensor):
            raise TypeError("Target resolver must return a torch.Tensor.")
        return score

    return _forward


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


def _mass_normalize_gradient(gradient: torch.Tensor, operators: MeshOperators) -> torch.Tensor:
    mass = require_mass(operators)
    n_vertices = infer_num_vertices_from_mass(mass)
    vertex_axis = _vertex_axis_from_feature_shape(tuple(gradient.shape), n_vertices)
    mass_vector = mass_to_vector(mass)

    if isinstance(mass_vector, torch.Tensor):
        mass_tensor = mass_vector.to(dtype=gradient.dtype, device=gradient.device)
    else:
        mass_tensor = torch.as_tensor(
            mass_vector,
            dtype=gradient.dtype,
            device=gradient.device,
        )

    if torch.any(mass_tensor <= 0):
        raise ValueError("Mass entries must be strictly positive for mass normalization.")

    reshape = [1] * gradient.ndim
    reshape[vertex_axis] = n_vertices
    return gradient / mass_tensor.reshape(reshape)


def _sum_over_channels(value: torch.Tensor, vertex_axis: int) -> torch.Tensor:
    channel_axes = [axis for axis in range(value.ndim) if axis > vertex_axis]
    if not channel_axes:
        return value

    reduced = value
    for axis in sorted(channel_axes, reverse=True):
        reduced = torch.sum(reduced, dim=axis)
    return reduced


def _multiply_mass(value: torch.Tensor, operators: MeshOperators) -> torch.Tensor:
    mass = require_mass(operators)
    n_vertices = infer_num_vertices_from_mass(mass)
    vertex_axis = _vertex_axis_from_feature_shape(tuple(value.shape), n_vertices)
    mass_vector = mass_to_vector(mass)

    if isinstance(mass_vector, torch.Tensor):
        mass_tensor = mass_vector.to(dtype=value.dtype, device=value.device)
    else:
        mass_tensor = torch.as_tensor(mass_vector, dtype=value.dtype, device=value.device)

    reshape = [1] * value.ndim
    reshape[vertex_axis] = n_vertices
    return value * mass_tensor.reshape(reshape)


def _mass_weighted_total(density_scalar: torch.Tensor, operators: MeshOperators) -> torch.Tensor:
    weighted = _multiply_mass(density_scalar, operators)
    return torch.sum(weighted)


def _target_to_metadata(target: object) -> object:
    if callable(target):
        fn = cast(Callable[[torch.Tensor], TensorLike], target)
        return f"callable:{getattr(fn, '__name__', 'anonymous')}"
    return target


def _vertex_axis_from_feature_shape(shape: tuple[int, ...], n_vertices: int) -> int:
    if len(shape) == 1:
        if shape[0] != n_vertices:
            raise ValueError(f"Expected shape [n] with n={n_vertices}, got {shape}.")
        return 0

    if len(shape) == 2:
        if shape[0] == n_vertices and shape[1] == n_vertices:
            return 0
        if shape[0] == n_vertices:
            return 0
        if shape[1] == n_vertices:
            return 1
        raise ValueError(f"Expected shape [n, C] with n={n_vertices}, got {shape}.")

    if len(shape) == 3:
        if shape[1] == n_vertices:
            return 1
        if shape[0] == n_vertices:
            return 0
        if shape[2] == n_vertices:
            return 2
        raise ValueError(f"Expected shape [B, n, C] with n={n_vertices}, got {shape}.")

    raise ValueError(
        f"Unsupported tensor rank {len(shape)} for shape {shape}. Expected [n,C] or [B,n,C]."
    )
