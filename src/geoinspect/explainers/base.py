"""Common base classes for explainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..operators import MeshOperators
from ..results import ExplanationResult
from ..types import MetadataDict, TargetLike, TensorLike


@dataclass(slots=True)
class ExplainerConfig:
    """Base configuration shared by explainers."""

    smooth: str | None = None
    normalize: str | None = None
    smooth_tau: float | None = None
    smooth_num_modes: int | None = None
    forward_kwargs: MetadataDict = field(default_factory=dict)
    prefer_operator_signature: bool = False


class BaseExplainer(ABC):
    """Abstract base class for all explainers."""

    def __init__(self, model: TensorLike, config: ExplainerConfig | None = None) -> None:
        self.model = model
        self.config = config or ExplainerConfig()

    @abstractmethod
    def explain(
        self,
        features: TensorLike,
        operators: MeshOperators,
        target: TargetLike,
    ) -> ExplanationResult:
        """Run explanation and return a structured result."""
