"""Result containers for explainability outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import MetadataDict, TensorLike


@dataclass(slots=True)
class ExplanationResult:
    """Structured output returned by explainers.

    Attributes:
        raw_map: Direct output from gradients/activations.
        density_map: Mesh-aware attribution per area unit.
        contribution_map: Area-weighted attribution map.
        smoothed_map: Optional post-processed map.
        metadata: Method and run information.
    """

    raw_map: TensorLike
    density_map: TensorLike
    contribution_map: TensorLike
    smoothed_map: TensorLike | None = None
    metadata: MetadataDict = field(default_factory=dict)
