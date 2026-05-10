"""Model forward adapters for generic and DiffusionNet-style signatures."""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import Signature, signature

from .operators import MeshOperators, coerce_operators
from .types import MetadataDict, TensorLike

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class ForwardConfig:
    """Forward-call configuration for explainers.

    Attributes:
        kwargs: Extra forward kwargs (e.g. ``faces`` for DiffusionNet).
        prefer_operator_signature: If true, skip plain ``model(features)`` and
            try operator-aware calls first.
    """

    kwargs: MetadataDict = field(default_factory=dict)
    prefer_operator_signature: bool = False


class ModelForwardError(RuntimeError):
    """Raised when model forward execution fails for all supported call styles."""


def run_model_forward(
    model: TensorLike,
    features: TensorLike,
    operators: MeshOperators | None = None,
    forward_config: ForwardConfig | None = None,
) -> TensorLike:
    """Run model forward with auto-adaptation for DiffusionNet signatures."""
    if not callable(model):
        raise TypeError("`model` must be callable.")

    config = forward_config or ForwardConfig()
    user_kwargs = dict(config.kwargs)

    first_error: Exception | None = None

    if user_kwargs and not config.prefer_operator_signature:
        try:
            return _validate_output(model(features, **user_kwargs))
        except Exception as exc:  # pragma: no cover - defensive path
            first_error = exc

    if not config.prefer_operator_signature:
        try:
            return _validate_output(model(features))
        except Exception as exc:
            first_error = exc

    if operators is None:
        if first_error is not None:
            raise ModelForwardError(
                "Model forward failed for plain call styles. Provide operators if "
                "your model expects DiffusionNet geometric inputs."
            ) from first_error
        raise ModelForwardError("Model forward failed and no operators were provided.")

    ops = coerce_operators(operators)

    keyword_error: Exception | None = None
    keyword_kwargs = _build_operator_kwargs(model, ops, user_kwargs)
    try:
        return _validate_output(model(features, **keyword_kwargs))
    except Exception as exc:
        keyword_error = exc

    positional_error: Exception | None = None
    try:
        positional_kwargs = {
            key: value
            for key, value in user_kwargs.items()
            if key
            not in {
                "mass",
                "L",
                "laplacian",
                "evals",
                "evecs",
                "gradX",
                "gradY",
                "grad_x",
                "grad_y",
            }
        }
        return _validate_output(
            model(
                features,
                ops.mass,
                ops.laplacian,
                ops.evals,
                ops.evecs,
                ops.grad_x,
                ops.grad_y,
                **positional_kwargs,
            )
        )
    except Exception as exc:
        positional_error = exc

    message = (
        "Model forward failed for all supported call styles: plain, keyword-operator, "
        "and positional DiffusionNet style."
    )
    if keyword_error is not None:
        message += f" Keyword error: {keyword_error!r}."
    if positional_error is not None:
        message += f" Positional error: {positional_error!r}."

    cause = positional_error or keyword_error or first_error
    raise ModelForwardError(message) from cause


def _build_operator_kwargs(
    model: TensorLike,
    operators: MeshOperators,
    user_kwargs: MetadataDict,
) -> MetadataDict:
    params = _signature_parameter_names(model)
    accepts_any_kwargs = _signature_accepts_varkw(model)

    candidate: MetadataDict = dict(user_kwargs)

    alias_values: tuple[tuple[str, TensorLike | None], ...] = (
        ("mass", operators.mass),
        ("L", operators.laplacian),
        ("laplacian", operators.laplacian),
        ("evals", operators.evals),
        ("evecs", operators.evecs),
        ("gradX", operators.grad_x),
        ("gradY", operators.grad_y),
        ("grad_x", operators.grad_x),
        ("grad_y", operators.grad_y),
    )

    for key, value in alias_values:
        if value is None:
            continue
        if key in candidate:
            continue
        if accepts_any_kwargs or key in params:
            candidate[key] = value

    return candidate


def _signature_parameter_names(model: TensorLike) -> set[str]:
    try:
        sig: Signature = signature(model)
    except (TypeError, ValueError):
        return set()
    return {param.name for param in sig.parameters.values()}


def _signature_accepts_varkw(model: TensorLike) -> bool:
    try:
        sig: Signature = signature(model)
    except (TypeError, ValueError):
        return True

    for param in sig.parameters.values():
        if param.kind == param.VAR_KEYWORD:
            return True
    return False


def _validate_output(output: TensorLike) -> TensorLike:
    if torch is not None and isinstance(output, torch.Tensor):
        return output
    if torch is not None and isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError("Model output must be a torch.Tensor, or a tuple/list containing one.")
