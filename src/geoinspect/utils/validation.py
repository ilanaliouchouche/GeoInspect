"""Validation helpers for user-facing APIs."""

from __future__ import annotations

from ..types import TensorLike


def require_non_none(name: str, value: TensorLike) -> TensorLike:
    """Ensure a required argument is present."""
    if value is None:
        raise ValueError(f"`{name}` must not be None.")
    return value
