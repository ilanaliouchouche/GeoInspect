"""Intrinsic Grad-CAM explainer."""

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
class GradCAMConfig(ExplainerConfig):
    """Configuration for intrinsic Grad-CAM."""

    target_layer: str = ""
    mass_weighted: bool = True
    use_relu: bool = True
    signed: bool = False
    return_channel_weights: bool = True


class IntrinsicGradCAMExplainer(BaseExplainer):
    """Compute Grad-CAM style explanations on meshes."""

    def __init__(self, model: TensorLike, config: GradCAMConfig | None = None) -> None:
        super().__init__(model=model, config=config or GradCAMConfig())

    def explain(
        self,
        features: TensorLike,
        operators: MeshOperators,
        target: TargetLike,
    ) -> ExplanationResult:
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for Intrinsic Grad-CAM.")
        if not isinstance(features, torch.Tensor):
            raise TypeError("`features` must be a torch.Tensor for Intrinsic Grad-CAM.")

        config = cast(GradCAMConfig, self.config)
        if not config.target_layer.strip():
            raise ValueError("`target_layer` must be a non-empty module path.")

        model = _as_module(self.model)
        target_module = _resolve_target_layer(model, config.target_layer)
        ops = coerce_operators(operators)
        forward_config = ForwardConfig(
            kwargs=dict(config.forward_kwargs),
            prefer_operator_signature=config.prefer_operator_signature,
        )

        feature_input = features.detach().clone()
        if not feature_input.is_floating_point():
            feature_input = feature_input.to(dtype=torch.float32)
        feature_input.requires_grad_(True)

        captured_activation: dict[str, torch.Tensor] = {}

        def _capture_activation(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: object,
        ) -> None:
            captured_activation["value"] = _extract_tensor_output(output)

        handle = target_module.register_forward_hook(_capture_activation)
        try:
            model_output = _call_model(
                model,
                feature_input,
                operators=ops,
                forward_config=forward_config,
            )
        finally:
            handle.remove()

        if "value" not in captured_activation:
            raise RuntimeError(
                f"No activation captured for layer '{config.target_layer}'. "
                "Ensure the layer is used during forward."
            )

        activation = captured_activation["value"]
        score = resolve_target(model_output, TargetSpec(target=target))

        activation_grads = torch.autograd.grad(
            score,
            activation,
            retain_graph=False,
            create_graph=False,
        )[0]

        n_vertices = infer_num_vertices_from_mass(require_mass(ops))
        vertex_axis, channel_axis = _vertex_and_channel_axes(tuple(activation.shape), n_vertices)

        channel_weights = _channel_importance_weights(
            gradients=activation_grads,
            operators=ops,
            vertex_axis=vertex_axis,
            channel_axis=channel_axis,
            mass_weighted=config.mass_weighted,
        )

        signed_cam = _combine_channels(
            activation=activation,
            channel_weights=channel_weights,
            channel_axis=channel_axis,
        )

        if config.signed:
            density_map = signed_cam
        elif config.use_relu:
            density_map = torch.relu(signed_cam)
        else:
            density_map = signed_cam

        if config.normalize not in {None, "", "none"}:
            density_map = cast(
                torch.Tensor,
                normalize_map(density_map, ops, method=config.normalize),
            )

        contribution_map = _multiply_mass(density_map, ops)

        if config.smooth in {None, "", "none"}:
            smoothed_map = None
        else:
            smooth_method = cast(str, config.smooth)
            smoothed_map = cast(
                torch.Tensor,
                smooth_map(
                    density_map,
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
            "method": "intrinsic_gradcam",
            "target_layer": config.target_layer,
            "mass_weighted": config.mass_weighted,
            "use_relu": config.use_relu,
            "signed": config.signed,
            "target": _target_to_metadata(target),
            "smoothing": config.smooth,
            "smooth_tau": config.smooth_tau,
            "smooth_num_modes": config.smooth_num_modes,
            "normalize": config.normalize,
            "score": float(score.detach().cpu().item()),
            "activation_shape": tuple(int(v) for v in activation.shape),
            "channel_axis": channel_axis,
            "vertex_axis": vertex_axis,
            "forward_kwargs_keys": sorted(config.forward_kwargs.keys()),
            "prefer_operator_signature": config.prefer_operator_signature,
        }
        if config.return_channel_weights:
            metadata["channel_weights"] = channel_weights.detach().clone()

        return ExplanationResult(
            raw_map=signed_cam.detach().clone(),
            density_map=density_map.detach().clone(),
            contribution_map=contribution_map.detach().clone(),
            smoothed_map=None if smoothed_map is None else smoothed_map.detach().clone(),
            metadata=metadata,
        )


def _as_module(model: TensorLike) -> torch.nn.Module:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("`model` must be an instance of `torch.nn.Module` for Grad-CAM.")
    return model


def _resolve_target_layer(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    layer_map = dict(model.named_modules())
    if layer_name not in layer_map:
        available = ", ".join(sorted(name for name in layer_map.keys() if name))
        raise ValueError(f"Unknown target layer '{layer_name}'. Available layers: {available}")
    return cast(torch.nn.Module, layer_map[layer_name])


def _extract_tensor_output(output: object) -> torch.Tensor:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Torch backend unavailable.")

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError("Target layer output must be a tensor or contain a tensor.")


def _call_model(
    model: torch.nn.Module,
    features: torch.Tensor,
    operators: MeshOperators,
    forward_config: ForwardConfig,
) -> torch.Tensor:
    output = run_model_forward(
        model,
        features,
        operators=operators,
        forward_config=forward_config,
    )
    if not isinstance(output, torch.Tensor):
        raise TypeError("Model output must be a torch.Tensor.")
    return output


def _channel_importance_weights(
    gradients: torch.Tensor,
    operators: MeshOperators,
    vertex_axis: int,
    channel_axis: int,
    mass_weighted: bool,
) -> torch.Tensor:
    if vertex_axis == channel_axis:
        raise ValueError("Vertex axis and channel axis cannot be identical.")

    grads_standard = torch.movedim(gradients, (vertex_axis, channel_axis), (0, -1))
    n_vertices = grads_standard.shape[0]

    if mass_weighted:
        mass = require_mass(operators)
        mass_vector = mass_to_vector(mass)
        if isinstance(mass_vector, torch.Tensor):
            mass_tensor = mass_vector.to(dtype=gradients.dtype, device=gradients.device)
        else:
            mass_tensor = torch.as_tensor(
                mass_vector,
                dtype=gradients.dtype,
                device=gradients.device,
            )

        if mass_tensor.ndim != 1 or mass_tensor.shape[0] != n_vertices:
            raise ValueError(
                f"Mass vector shape mismatch: expected [{n_vertices}], "
                f"got {tuple(mass_tensor.shape)}."
            )

        weighted = grads_standard * mass_tensor.reshape(
            n_vertices,
            *([1] * (grads_standard.ndim - 1)),
        )
        numerator = torch.sum(weighted, dim=0)
        denominator = torch.sum(mass_tensor)
        if float(denominator.detach().cpu().item()) <= 0.0:
            raise ValueError("Mass vector must have strictly positive sum.")
        return numerator / denominator

    return torch.mean(grads_standard, dim=0)


def _combine_channels(
    activation: torch.Tensor,
    channel_weights: torch.Tensor,
    channel_axis: int,
) -> torch.Tensor:
    activation_with_channels_last = torch.movedim(activation, channel_axis, -1)

    if channel_weights.ndim == 1:
        weighted = activation_with_channels_last * channel_weights.reshape(
            *([1] * (activation_with_channels_last.ndim - 1)),
            -1,
        )
        return torch.sum(weighted, dim=-1)

    if channel_weights.ndim == 2:
        if activation_with_channels_last.ndim != 3:
            raise ValueError(
                "Batch-wise channel weights require activation rank 3 with channel axis."
            )
        weighted = activation_with_channels_last * channel_weights.reshape(
            channel_weights.shape[0],
            1,
            channel_weights.shape[1],
        )
        return torch.sum(weighted, dim=-1)

    raise ValueError("Unsupported channel weights rank for Grad-CAM combination.")


def _multiply_mass(value: torch.Tensor, operators: MeshOperators) -> torch.Tensor:
    mass = require_mass(operators)
    n_vertices = infer_num_vertices_from_mass(mass)
    vertex_axis, _ = _vertex_and_channel_axes(
        tuple(value.shape),
        n_vertices,
        require_channel_axis=False,
    )
    mass_vector = mass_to_vector(mass)

    if isinstance(mass_vector, torch.Tensor):
        mass_tensor = mass_vector.to(dtype=value.dtype, device=value.device)
    else:
        mass_tensor = torch.as_tensor(mass_vector, dtype=value.dtype, device=value.device)

    reshape = [1] * value.ndim
    reshape[vertex_axis] = n_vertices
    return value * mass_tensor.reshape(reshape)


def _target_to_metadata(target: object) -> object:
    if callable(target):
        fn = cast(Callable[[torch.Tensor], TensorLike], target)
        return f"callable:{getattr(fn, '__name__', 'anonymous')}"
    return target


def _vertex_and_channel_axes(
    shape: tuple[int, ...],
    n_vertices: int,
    require_channel_axis: bool = True,
) -> tuple[int, int]:
    if len(shape) == 1:
        if shape[0] != n_vertices:
            raise ValueError(f"Expected shape [n] with n={n_vertices}, got {shape}.")
        if require_channel_axis:
            raise ValueError("Grad-CAM activations need a channel axis.")
        return 0, -1

    if len(shape) == 2:
        if shape[0] == n_vertices and shape[1] == n_vertices:
            return 0, 1
        if shape[0] == n_vertices:
            return 0, 1
        if shape[1] == n_vertices:
            return 1, 0
        raise ValueError(f"Expected shape [n, C] or [C, n] with n={n_vertices}, got {shape}.")

    if len(shape) == 3:
        if shape[1] == n_vertices:
            return 1, 2
        if shape[2] == n_vertices:
            return 2, 1
        if shape[0] == n_vertices:
            return 0, 1
        raise ValueError(
            f"Expected shape [B, n, C] (or equivalent) with n={n_vertices}, got {shape}."
        )

    raise ValueError(
        f"Unsupported tensor rank {len(shape)} for shape {shape}. Expected [n,C] or [B,n,C]."
    )
