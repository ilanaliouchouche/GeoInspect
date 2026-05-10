"""Shared typing aliases used across GeoInspect."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

TensorLike: TypeAlias = Any
TargetFn: TypeAlias = Callable[[Any], Any]
TargetLike: TypeAlias = int | tuple[int, ...] | None | TargetFn
MetadataDict: TypeAlias = dict[str, Any]
