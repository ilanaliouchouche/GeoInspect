"""Target selection helpers for scalar objectives."""

from __future__ import annotations

from dataclasses import dataclass

from .types import TargetFn, TargetLike, TensorLike

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class TargetSpec:
    """Target definition for explanation calls."""

    target: TargetLike = None
    target_fn: TargetFn | None = None


class TargetResolutionError(RuntimeError):
    """Raised when a target cannot be resolved to a scalar objective."""


def resolve_target(model_output: TensorLike, spec: TargetSpec | None) -> TensorLike:
    """Resolve the scalar objective from model output.

    Returns a scalar tensor suitable for autograd.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for target resolution in gradient explainers.")
    if not isinstance(model_output, torch.Tensor):
        raise TypeError("`model_output` must be a torch.Tensor.")

    target_spec = spec or TargetSpec()
    target_fn = target_spec.target_fn
    target = target_spec.target

    if callable(target):
        if target_fn is not None:
            raise TargetResolutionError(
                "Provide either `target` callable or `target_fn`, not both."
            )
        target_fn = target
        target = None

    if target_fn is not None:
        score = target_fn(model_output)
        if not isinstance(score, torch.Tensor):
            score = torch.as_tensor(score, dtype=model_output.dtype, device=model_output.device)
        return _as_scalar_score(score)

    if target is None:
        if model_output.numel() != 1:
            raise TargetResolutionError(
                "`target` is required when model output has more than one element."
            )
        return model_output.reshape(())

    if isinstance(target, int):
        if model_output.ndim == 0:
            raise TargetResolutionError("Integer `target` requires non-scalar model outputs.")
        try:
            selected = model_output[..., int(target)]
        except Exception as exc:  # pragma: no cover - defensive
            raise TargetResolutionError(
                f"Failed to index model output with target={target}."
            ) from exc
        return _as_scalar_score(selected)

    if isinstance(target, tuple):
        if len(target) == 0:
            raise TargetResolutionError("Tuple target must contain at least one index.")
        if not all(isinstance(idx, int) for idx in target):
            raise TargetResolutionError("Tuple target must contain only integer indices.")
        try:
            selected = model_output[tuple(int(idx) for idx in target)]
        except Exception as exc:  # pragma: no cover - defensive
            raise TargetResolutionError(
                f"Failed to index model output with tuple target={target}."
            ) from exc
        return _as_scalar_score(selected)

    raise TargetResolutionError(
        "`target` must be None, int, tuple[int, ...], or a callable target function."
    )


def _as_scalar_score(value: TensorLike) -> TensorLike:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch backend unavailable.")
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected a torch.Tensor.")
    if value.ndim == 0:
        return value
    if value.numel() == 1:
        return value.reshape(())
    return value.sum()
