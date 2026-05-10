"""GeoInspect public package API."""

from ._version import __version__
from .baselines import BaselineConfig, build_baseline
from .evaluation import (
    CheckResult,
    check_constant_signal,
    check_ig_completeness,
    check_mass_consistency,
    check_smoothing_energy,
)
from .explainers import (
    BaseExplainer,
    ExplainerConfig,
    GradCAMConfig,
    IntegratedGradientsConfig,
    IntegratedGradientsExplainer,
    IntrinsicGradCAMExplainer,
    SaliencyConfig,
    SaliencyExplainer,
)
from .forward import ForwardConfig, ModelForwardError, run_model_forward
from .normalization import NormalizationConfig, normalize_map
from .operators import MeshOperators
from .results import ExplanationResult
from .smoothing import SmoothingConfig, smooth_map
from .targets import TargetResolutionError, TargetSpec, resolve_target
from .visualization import (
    PolyscopeViewerConfig,
    launch_polyscope_viewer,
    prepare_vertex_scalar_map,
)

__all__ = [
    "BaseExplainer",
    "BaselineConfig",
    "CheckResult",
    "ExplainerConfig",
    "ExplanationResult",
    "ForwardConfig",
    "GradCAMConfig",
    "IntegratedGradientsConfig",
    "IntegratedGradientsExplainer",
    "IntrinsicGradCAMExplainer",
    "MeshOperators",
    "ModelForwardError",
    "NormalizationConfig",
    "PolyscopeViewerConfig",
    "SaliencyConfig",
    "SaliencyExplainer",
    "SmoothingConfig",
    "TargetResolutionError",
    "TargetSpec",
    "__version__",
    "build_baseline",
    "check_constant_signal",
    "check_ig_completeness",
    "check_mass_consistency",
    "check_smoothing_energy",
    "launch_polyscope_viewer",
    "normalize_map",
    "prepare_vertex_scalar_map",
    "resolve_target",
    "run_model_forward",
    "smooth_map",
]
