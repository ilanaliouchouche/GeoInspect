"""Polyscope visualization helpers for mesh-aware attribution maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from .operators import (
    MeshOperators,
    coerce_operators,
    infer_vertex_axis,
    is_torch_tensor,
)
from .smoothing import smooth_map
from .types import TensorLike


@dataclass(slots=True)
class PolyscopeViewerConfig:
    """Runtime options for interactive Polyscope visualization."""

    mesh_name: str = "geoinspect_mesh"
    quantity_name: str = "attribution"
    cmap: str = "coolwarm"
    collapse: str = "l2"
    smoothing_method: str = "none"
    initial_tau: float = 1e-2
    tau_min: float = 0.0
    tau_max: float = 0.25
    num_modes: int | None = None
    enabled: bool = True
    smooth_shade: bool = True
    show: bool = True


def prepare_vertex_scalar_map(
    scalar_map: TensorLike,
    n_vertices: int,
    collapse: str = "l2",
) -> np.ndarray:
    """Convert a map of shape ``[n,*]`` or ``[*,n,*]`` to scalar values ``[n]``."""
    if n_vertices <= 0:
        raise ValueError("`n_vertices` must be strictly positive.")

    values = np.asarray(_to_numpy(scalar_map), dtype=np.float64)
    axis = infer_vertex_axis(values.shape, n_vertices)
    moved = np.moveaxis(values, axis, 0)
    flattened = moved.reshape(n_vertices, -1)
    return _collapse_columns(flattened, collapse=collapse)


def launch_polyscope_viewer(
    vertices: TensorLike,
    faces: TensorLike,
    scalar_map: TensorLike,
    *,
    operators: MeshOperators | None = None,
    config: PolyscopeViewerConfig | None = None,
) -> None:
    """Launch Polyscope viewer with optional interactive smoothing slider."""
    cfg = config or PolyscopeViewerConfig()
    vertices_np = np.asarray(_to_numpy(vertices), dtype=np.float64)
    faces_np = np.asarray(_to_numpy(faces), dtype=np.int64)

    if vertices_np.ndim != 2 or vertices_np.shape[1] != 3:
        raise ValueError("`vertices` must have shape [n_vertices, 3].")
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError("`faces` must have shape [n_faces, 3].")

    n_vertices = int(vertices_np.shape[0])
    base_map = prepare_vertex_scalar_map(scalar_map, n_vertices=n_vertices, collapse=cfg.collapse)
    ops = None if operators is None else coerce_operators(operators)

    try:
        import polyscope as ps
        import polyscope.imgui as psim
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Polyscope is required for interactive visualization. "
            "Install with `pip install geoinspect[viz]`."
        ) from exc
    ps_mod = cast(Any, ps)
    psim_mod = cast(Any, psim)

    is_initialized = False
    if hasattr(ps, "is_initialized"):
        is_initialized = bool(ps_mod.is_initialized())
    if not is_initialized:
        ps_mod.init()

    mesh = ps_mod.register_surface_mesh(
        cfg.mesh_name,
        vertices_np,
        faces_np,
        smooth_shade=cfg.smooth_shade,
    )

    initial_values = _maybe_smooth_map(
        base_map,
        operators=ops,
        method=cfg.smoothing_method,
        tau=cfg.initial_tau,
        num_modes=cfg.num_modes,
    )
    mesh.add_scalar_quantity(
        cfg.quantity_name,
        initial_values,
        enabled=cfg.enabled,
        cmap=cfg.cmap,
    )

    state: dict[str, float] = {"tau": float(cfg.initial_tau)}
    method_name = cfg.smoothing_method.strip().lower()

    def _refresh_quantity(updated_values: np.ndarray) -> None:
        mesh.add_scalar_quantity(
            cfg.quantity_name,
            updated_values,
            enabled=cfg.enabled,
            cmap=cfg.cmap,
        )

    def _callback() -> None:
        if method_name in {"", "none"} or ops is None:
            return

        changed, tau_value = psim_mod.SliderFloat(
            "t (tau)",
            state["tau"],
            float(cfg.tau_min),
            float(cfg.tau_max),
        )
        if not changed:
            return

        state["tau"] = float(tau_value)
        updated = _maybe_smooth_map(
            base_map,
            operators=ops,
            method=method_name,
            tau=state["tau"],
            num_modes=cfg.num_modes,
        )
        _refresh_quantity(updated)

    ps_mod.set_user_callback(_callback)
    if cfg.show:
        ps_mod.show()


def _maybe_smooth_map(
    base_map: np.ndarray,
    *,
    operators: MeshOperators | None,
    method: str,
    tau: float,
    num_modes: int | None,
) -> np.ndarray:
    method_name = method.strip().lower()
    if method_name in {"", "none"} or operators is None:
        return cast(np.ndarray, np.asarray(base_map, dtype=np.float64).copy())

    smoothed = smooth_map(
        base_map,
        operators=operators,
        method=method_name,
        tau=tau,
        num_modes=num_modes,
    )
    return cast(np.ndarray, np.asarray(_to_numpy(smoothed), dtype=np.float64).reshape(-1))


def _collapse_columns(values: np.ndarray, collapse: str) -> np.ndarray:
    mode = collapse.strip().lower()
    if mode == "mean":
        return np.asarray(np.mean(values, axis=1), dtype=np.float64)
    if mode == "l2":
        return np.asarray(np.sqrt(np.sum(values * values, axis=1)), dtype=np.float64)
    if mode == "max_abs":
        return np.asarray(np.max(np.abs(values), axis=1), dtype=np.float64)
    if mode == "first":
        return np.asarray(values[:, 0], dtype=np.float64)
    raise ValueError("`collapse` must be one of: mean, l2, max_abs, first.")


def _to_numpy(value: TensorLike) -> np.ndarray:
    if is_torch_tensor(value):
        return cast(np.ndarray, value.detach().cpu().numpy())
    return np.asarray(value)
