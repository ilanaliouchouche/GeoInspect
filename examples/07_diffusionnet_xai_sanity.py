"""End-to-end GeoInspect sanity check on a trained DiffusionNet checkpoint.

What this script validates:
1. Checkpoint/model compatibility (rebuild + load + forward pass)
2. GeoInspect explainers execution (Saliency, IG, Intrinsic Grad-CAM)
3. Mathematical sanity checks (mass, constant signal, IG completeness, smoothing energy)
4. Optional Polyscope visual inspection with interactive tau slider.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import potpourri3d as pp3d
import torch

try:
    import diffusion_net

    from geoinspect import (
        ForwardConfig,
        GradCAMConfig,
        IntegratedGradientsConfig,
        IntegratedGradientsExplainer,
        IntrinsicGradCAMExplainer,
        MeshOperators,
        PolyscopeViewerConfig,
        SaliencyConfig,
        SaliencyExplainer,
        check_constant_signal,
        check_ig_completeness,
        check_mass_consistency,
        check_smoothing_energy,
        launch_polyscope_viewer,
        prepare_vertex_scalar_map,
        run_model_forward,
        smooth_map,
    )
except ImportError:
    repo_root_fallback = Path(__file__).resolve().parents[1]
    src_fallback = repo_root_fallback / "src"
    diffnet_src_fallback = repo_root_fallback / "third_party" / "diffusion-net" / "src"

    if str(src_fallback) not in sys.path:
        sys.path.insert(0, str(src_fallback))
    if str(diffnet_src_fallback) not in sys.path:
        sys.path.insert(0, str(diffnet_src_fallback))

    import diffusion_net

    from geoinspect import (
        ForwardConfig,
        GradCAMConfig,
        IntegratedGradientsConfig,
        IntegratedGradientsExplainer,
        IntrinsicGradCAMExplainer,
        MeshOperators,
        PolyscopeViewerConfig,
        SaliencyConfig,
        SaliencyExplainer,
        check_constant_signal,
        check_ig_completeness,
        check_mass_consistency,
        check_smoothing_energy,
        launch_polyscope_viewer,
        prepare_vertex_scalar_map,
        run_model_forward,
        smooth_map,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class SampleMesh:
    vertices: torch.Tensor
    faces: torch.Tensor
    class_name: str
    source_path: str


@dataclass(slots=True)
class ViewerMesh:
    mesh_name: str
    vertices: torch.Tensor
    faces: torch.Tensor
    maps: dict[str, torch.Tensor]
    operators: MeshOperators


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GeoInspect sanity check on DiffusionNet SHREC11")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to best_xai.pt or *.ckpt"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="",
        choices=["", "simplified", "original"],
        help="Override dataset type (default: infer from checkpoint)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="",
        help="Override dataset root (default: infer from checkpoint / SHREC11 experiment paths)",
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        default="",
        help="Optional direct mesh path (.obj/.off). If set, class selection is bypassed.",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="",
        help=(
            "Preferred class, or comma-separated classes when auto-selecting meshes "
            "(default: all checkpoint classes if <=2, otherwise first class)"
        ),
    )
    parser.add_argument(
        "--meshes_per_label",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of meshes to show per label (1 or 2)",
    )
    parser.add_argument(
        "--num_meshes",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--k_eig",
        type=int,
        default=0,
        help="Number of eigenpairs for operators (0 -> infer from checkpoint, else default 128)",
    )
    parser.add_argument(
        "--input_features",
        type=str,
        default="",
        choices=["", "xyz", "hks"],
        help="Override feature type (default: infer from checkpoint)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device for XAI run",
    )
    parser.add_argument("--ig_steps", type=int, default=24, help="Integrated gradients steps")
    parser.add_argument(
        "--target_layer",
        type=str,
        default="",
        help="Grad-CAM target layer (default: auto from last block)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/diffusionnet_xai_sanity",
        help="Directory to store reports and maps",
    )
    parser.add_argument(
        "--show_polyscope", action="store_true", help="Open interactive Polyscope GUI"
    )
    parser.add_argument(
        "--viewer_method",
        type=str,
        default="heat",
        choices=["none", "heat", "helmholtz"],
        help="Smoothing method in viewer",
    )
    parser.add_argument(
        "--viewer_tau", type=float, default=0.02, help="Initial tau for viewer slider"
    )
    parser.add_argument(
        "--viewer_tau_max", type=float, default=0.2, help="Max tau for viewer slider"
    )
    parser.add_argument(
        "--viewer_map",
        type=str,
        default="ig",
        choices=["ig", "saliency", "gradcam"],
        help="Attribution map to display in Polyscope",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    req = device_arg.strip().lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested 'cuda' but CUDA is unavailable.")
        return torch.device("cuda:0")
    if req == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Requested 'mps' but MPS is unavailable.")
        return torch.device("mps")
    if req == "cpu":
        return torch.device("cpu")
    raise ValueError("Unsupported device value.")


def _safe_torch_load(path: str) -> dict[str, Any]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")

    if isinstance(loaded, dict):
        return loaded
    raise TypeError("Checkpoint payload must be a dict.")


def _jsonable(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _get_model_state_dict(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        return ckpt["model_state_dict"]

    # Fallback: plain state_dict checkpoints.
    tensor_items = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    if "first_lin.weight" in tensor_items and "last_lin.weight" in tensor_items:
        return tensor_items

    raise ValueError("Could not locate model_state_dict in checkpoint payload.")


def _infer_model_config(
    state_dict: dict[str, torch.Tensor], ckpt: dict[str, Any]
) -> dict[str, Any]:
    cfg = ckpt.get("model_config", {})
    if not isinstance(cfg, dict):
        cfg = {}

    if "first_lin.weight" not in state_dict or "last_lin.weight" not in state_dict:
        raise ValueError("State dict missing first_lin/last_lin weights.")

    first_w = state_dict["first_lin.weight"]
    last_w = state_dict["last_lin.weight"]

    if first_w.ndim != 2 or last_w.ndim != 2:
        raise ValueError("Unexpected first_lin/last_lin weight shapes.")

    c_width = int(first_w.shape[0])
    c_in = int(first_w.shape[1])
    c_out = int(last_w.shape[0])

    max_block = -1
    for key in state_dict.keys():
        match = re.match(r"block_(\d+)\.", key)
        if match:
            max_block = max(max_block, int(match.group(1)))
    n_block = max_block + 1 if max_block >= 0 else int(cfg.get("N_block", 4))

    out: dict[str, Any] = {
        "C_in": int(cfg.get("C_in", c_in)),
        "C_out": int(cfg.get("C_out", c_out)),
        "C_width": int(cfg.get("C_width", c_width)),
        "N_block": int(cfg.get("N_block", n_block)),
        "outputs_at": str(cfg.get("outputs_at", "global_mean")),
        "dropout": bool(cfg.get("dropout", False)),
        "last_activation": str(cfg.get("last_activation", "log_softmax")),
    }
    return out


def _build_diffusionnet_model(model_cfg: dict[str, Any]) -> torch.nn.Module:
    def _last_activation_log_softmax(value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.log_softmax(value, dim=-1)

    activation_name = str(model_cfg.get("last_activation", "log_softmax")).strip().lower()
    if activation_name in {"", "none"}:
        last_activation = None
    elif activation_name == "log_softmax":
        last_activation = _last_activation_log_softmax
    else:
        raise ValueError(f"Unsupported last_activation '{activation_name}'.")

    model = diffusion_net.layers.DiffusionNet(
        C_in=int(model_cfg["C_in"]),
        C_out=int(model_cfg["C_out"]),
        C_width=int(model_cfg["C_width"]),
        N_block=int(model_cfg["N_block"]),
        outputs_at=str(model_cfg["outputs_at"]),
        dropout=bool(model_cfg["dropout"]),
        last_activation=last_activation,
    )
    return model


def _parse_original_categories(root_dir: Path) -> dict[str, list[str]]:
    cat_path = root_dir / "categories.txt"
    if not cat_path.exists():
        raise FileNotFoundError(f"Missing categories file: {cat_path}")

    classes: dict[str, list[str]] = {}
    with cat_path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    cursor = 2
    for _ in range(30):
        cursor += 1
        header = lines[cursor].strip().split()
        cursor += 1
        if len(header) != 3:
            raise ValueError("Unexpected categories.txt format.")
        class_name = header[0]
        mesh_ids = []
        for _i in range(20):
            mesh_ids.append(lines[cursor].strip())
            cursor += 1
        classes[class_name] = mesh_ids

    return classes


def _parse_class_names_arg(raw_value: str) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for part in raw_value.split(","):
        value = part.strip()
        if value and value not in seen:
            labels.append(value)
            seen.add(value)
    return labels


def _load_sample_mesh(path: Path, class_name: str) -> SampleMesh:
    if not path.exists():
        raise FileNotFoundError(f"Mesh path not found: {path}")
    verts_np, faces_np = pp3d.read_mesh(str(path))
    return SampleMesh(
        vertices=torch.tensor(verts_np, dtype=torch.float32),
        faces=torch.tensor(faces_np, dtype=torch.long),
        class_name=class_name,
        source_path=str(path),
    )


def _select_sample_meshes(
    dataset_type: str,
    dataset_root: Path,
    class_names: list[str],
    mesh_path: str,
    meshes_per_label: int,
) -> list[SampleMesh]:
    if meshes_per_label not in {1, 2}:
        raise ValueError("`meshes_per_label` must be 1 or 2.")

    if mesh_path.strip():
        path = Path(mesh_path).expanduser().resolve()
        guessed_class = class_names[0] if class_names else path.parent.name
        return [_load_sample_mesh(path, guessed_class)]

    samples: list[SampleMesh] = []
    if dataset_type == "simplified":
        for class_name in class_names:
            raw_root = dataset_root / "raw" / "shrec_16" / class_name
            if not raw_root.exists():
                raise FileNotFoundError(
                    f"Class folder not found for simplified dataset: {raw_root}"
                )
            candidates = sorted((raw_root / "train").glob("*.obj")) + sorted(
                (raw_root / "test").glob("*.obj")
            )
            if len(candidates) < meshes_per_label:
                raise FileNotFoundError(
                    f"Need {meshes_per_label} meshes for class '{class_name}', "
                    f"found {len(candidates)} in {raw_root}/train and /test"
                )
            #samples.append(_load_sample_mesh(candidates[12], class_name))
            #samples.append(_load_sample_mesh(candidates[18], class_name))
            
            for chosen in candidates[:meshes_per_label]:
                pass
                samples.append(_load_sample_mesh(chosen, class_name))
        return samples

    if dataset_type == "original":
        classes = _parse_original_categories(dataset_root)
        for class_name in class_names:
            if class_name not in classes:
                raise ValueError(
                    f"Class '{class_name}' not found in categories.txt. "
                    f"Available: {', '.join(sorted(classes.keys()))}"
                )
            mesh_ids = classes[class_name]
            start = 5 if len(mesh_ids) >= (5 + meshes_per_label) else 0
            selected_ids = mesh_ids[start : start + meshes_per_label]
            if len(selected_ids) < meshes_per_label:
                raise FileNotFoundError(
                    f"Need {meshes_per_label} mesh ids for class '{class_name}', "
                    f"found {len(selected_ids)}."
                )
            for mesh_id in selected_ids:
                pass
                #chosen = dataset_root / "raw" / f"T{mesh_id}.off"
                #samples.append(_load_sample_mesh(chosen, class_name))
            
            samples.append(_load_sample_mesh(dataset_root / "raw" / f"T{mesh_ids[5]}.off", class_name))
            samples.append(_load_sample_mesh(dataset_root / "raw" / f"T{mesh_ids[3]}.off", class_name))
        return samples

    raise ValueError("Unsupported dataset type.")


def _build_hks_features(evals: torch.Tensor, evecs: torch.Tensor, count: int = 16) -> torch.Tensor:
    if evals.device.type == "mps":
        scales = torch.logspace(-2, 0.0, steps=count, device="cpu", dtype=evals.dtype).to(
            evals.device
        )
        return diffusion_net.geometry.compute_hks(evals, evecs, scales)
    return diffusion_net.geometry.compute_hks_autoscale(evals, evecs, count)


def _to_device(
    tensor: torch.Tensor, device: torch.device, make_dense: bool = False
) -> torch.Tensor:
    out = tensor
    if make_dense and out.is_sparse:
        out = out.coalesce().to_dense()
    return out.to(device)


def _resolve_sample_target_label(
    sample_class_name: str,
    class_names: list[str],
    n_classes: int,
) -> tuple[int, str, str]:
    if not class_names:
        raise ValueError(
            "Checkpoint has no class_names; cannot assign per-mesh target from class label."
        )
    if sample_class_name not in class_names:
        available = ", ".join(class_names)
        raise ValueError(
            f"Mesh class '{sample_class_name}' is not in checkpoint classes: {available}"
        )
    target_idx = class_names.index(sample_class_name)
    if target_idx < 0 or target_idx >= n_classes:
        raise ValueError(
            f"Resolved target index {target_idx} out of range for {n_classes} classes."
        )
    return target_idx, class_names[target_idx], "sample_class_name"


def _launch_polyscope_method_picker(
    *,
    viewer_meshes: list[ViewerMesh],
    initial_map: str,
    initial_method: str,
    initial_tau: float,
    tau_min: float,
    tau_max: float,
) -> None:
    try:
        import polyscope as ps
        import polyscope.imgui as psim
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Polyscope is required for interactive visualization. "
            "Install with `pip install geoinspect[viz]`."
        ) from exc

    map_keys = ["ig", "saliency", "gradcam"]
    if not viewer_meshes:
        raise ValueError("`viewer_meshes` must not be empty.")

    method_candidates = ["none", "heat", "helmholtz"]
    method_name = initial_method.strip().lower()
    if method_name not in method_candidates:
        method_name = "heat"

    map_name = initial_map.strip().lower()
    if map_name not in map_keys:
        map_name = "ig"

    if not ps.is_initialized():
        ps.init()
        ps.set_navigation_style("free")
        ps.set_ground_plane_mode("none")

    registered: list[tuple[Any, dict[str, np.ndarray], MeshOperators]] = []
    layout_gap = 0.15

    # Build a layout axis orthogonal to the dominant geometric axis of the first mesh.
    first_vertices = viewer_meshes[0].vertices.detach().cpu().numpy()
    first_centered = first_vertices - np.mean(first_vertices, axis=0, keepdims=True)
    cov = first_centered.T @ first_centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, int(np.argmax(eigvals))]
    basis = np.eye(3, dtype=np.float64)
    orth_axis_idx = int(np.argmin(np.abs(basis @ principal_axis)))
    layout_axis = basis[orth_axis_idx]
    layout_axis = layout_axis / max(np.linalg.norm(layout_axis), 1e-12)

    cursor = 0.0
    for index, item in enumerate(viewer_meshes):
        vertices_np = item.vertices.detach().cpu().numpy()
        faces_np = item.faces.detach().cpu().numpy()
        n_vertices = int(vertices_np.shape[0])
        centered_vertices = vertices_np - np.mean(vertices_np, axis=0, keepdims=True)
        proj = centered_vertices @ layout_axis
        p_min = float(np.min(proj)) if n_vertices > 0 else 0.0
        p_max = float(np.max(proj)) if n_vertices > 0 else 0.0
        width = max(p_max - p_min, 1e-6)
        offset = cursor - p_min
        cursor += width + layout_gap * width

        shifted_vertices = centered_vertices + (offset * layout_axis.reshape(1, 3))
        mesh = ps.register_surface_mesh(item.mesh_name, shifted_vertices, faces_np, smooth_shade=True)

        scalar_maps_np = {
            key: prepare_vertex_scalar_map(
                item.maps[key].detach().cpu(),
                n_vertices=n_vertices,
                collapse="first",
            )
            for key in map_keys
        }
        registered.append((mesh, scalar_maps_np, item.operators))

    state: dict[str, object] = {
        "map": map_name,
        "method": method_name,
        "tau": float(initial_tau),
    }

    def _refresh() -> None:
        current_map = str(state["map"])
        current_method = str(state["method"])
        tau_value = float(state["tau"])
        for mesh, scalar_maps_np, operators in registered:
            values = scalar_maps_np[current_map]
            if current_method != "none":
                values = np.asarray(
                    smooth_map(values, operators=operators, method=current_method, tau=tau_value),
                    dtype=np.float64,
                ).reshape(-1)
            quantity_name = f"{current_map}_density ({current_method})"
            mesh.add_scalar_quantity(quantity_name, values, enabled=True, cmap="coolwarm")

    _refresh()

    def _callback() -> None:
        dirty = False

        if psim.Button("Map: IG"):
            state["map"] = "ig"
            dirty = True
        if psim.Button("Map: Saliency"):
            state["map"] = "saliency"
            dirty = True
        if psim.Button("Map: GradCAM"):
            state["map"] = "gradcam"
            dirty = True

        if psim.Button("Smooth: none"):
            state["method"] = "none"
            dirty = True
        if psim.Button("Smooth: heat"):
            state["method"] = "heat"
            dirty = True
        if psim.Button("Smooth: helmholtz"):
            state["method"] = "helmholtz"
            dirty = True

        changed, tau_value = psim.SliderFloat("t (tau)", float(state["tau"]), tau_min, tau_max)
        if changed:
            state["tau"] = float(tau_value)
            dirty = True

        if dirty:
            _refresh()

    ps.set_user_callback(_callback)
    ps.show()


def _auto_target_layer(model: torch.nn.Module, user_layer: str) -> str:
    layer_map = dict(model.named_modules())
    if user_layer.strip():
        if user_layer not in layer_map:
            available = ", ".join(sorted(name for name in layer_map.keys() if name))
            raise ValueError(f"Unknown --target_layer '{user_layer}'. Available: {available}")
        return user_layer

    n_block = int(getattr(model, "N_block", 0))
    candidates = [
        f"block_{max(n_block - 1, 0)}.mlp",
        f"block_{max(n_block - 1, 0)}",
        "last_lin",
        "first_lin",
    ]
    for cand in candidates:
        if cand in layer_map:
            return cand

    non_empty = [name for name in layer_map.keys() if name]
    if not non_empty:
        raise ValueError("No named modules found for Grad-CAM layer selection.")
    return non_empty[-1]


def _prepare_mesh_operators(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    k_eig: int,
    op_cache_dir: Path,
    device: torch.device,
    diffusion_method: str,
) -> tuple[MeshOperators, MeshOperators, torch.Tensor, torch.Tensor]:
    verts = diffusion_net.geometry.normalize_positions(vertices)
    faces = faces.long()

    frames, mass, laplacian, evals, evecs, grad_x, grad_y = diffusion_net.geometry.get_operators(
        verts,
        faces,
        k_eig=k_eig,
        op_cache_dir=str(op_cache_dir),
    )
    del frames

    laplacian_checks = laplacian.coalesce().to_dense() if laplacian.is_sparse else laplacian

    ops_checks = MeshOperators(
        mass=mass,
        stiffness=laplacian_checks,
        laplacian=laplacian_checks,
        evals=evals,
        evecs=evecs,
        grad_x=grad_x,
        grad_y=grad_y,
    )

    dense_for_mps = device.type == "mps"
    laplacian_model = laplacian
    if diffusion_method == "spectral" and device.type == "mps":
        # Spectral mode does not need L in DiffusionNet and MPS sparse support is partial.
        laplacian_model = None

    ops_model = MeshOperators(
        mass=_to_device(mass, device),
        stiffness=None
        if laplacian_model is None
        else _to_device(laplacian_model, device, make_dense=dense_for_mps),
        laplacian=None
        if laplacian_model is None
        else _to_device(laplacian_model, device, make_dense=dense_for_mps),
        evals=_to_device(evals, device),
        evecs=_to_device(evecs, device),
        grad_x=_to_device(grad_x, device, make_dense=dense_for_mps),
        grad_y=_to_device(grad_y, device, make_dense=dense_for_mps),
    )

    return ops_model, ops_checks, _to_device(verts, device), _to_device(faces, device)


def main() -> int:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    ckpt = _safe_torch_load(str(checkpoint_path))
    state_dict = _get_model_state_dict(ckpt)
    model_cfg = _infer_model_config(state_dict, ckpt)

    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt.get("args", {}), dict) else {}
    ckpt_dataset = ckpt.get("dataset", {}) if isinstance(ckpt.get("dataset", {}), dict) else {}

    dataset_type = args.dataset_type or str(
        ckpt_args.get("dataset_type", ckpt_dataset.get("dataset_type", "simplified"))
    )
    input_features = args.input_features or str(ckpt_args.get("input_features", "hks"))

    if dataset_type not in {"simplified", "original"}:
        raise ValueError(f"Unsupported dataset_type '{dataset_type}'.")
    if input_features not in {"xyz", "hks"}:
        raise ValueError(f"Unsupported input_features '{input_features}'.")

    if args.dataset_root.strip():
        dataset_root = Path(args.dataset_root).expanduser().resolve()
    else:
        inferred = ckpt_dataset.get("dataset_path", "")
        if inferred:
            dataset_root = Path(str(inferred)).expanduser().resolve()
        else:
            dataset_root = (
                REPO_ROOT
                / "third_party"
                / "diffusion-net"
                / "experiments"
                / "classification_shrec11"
                / "data"
                / dataset_type
            ).resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}. Provide --dataset_root explicitly."
        )

    class_names = ckpt.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        class_names = ckpt_dataset.get("class_names", [])
    if not isinstance(class_names, list):
        class_names = []
    class_names = [str(name).strip() for name in class_names if str(name).strip()]

    requested_classes = _parse_class_names_arg(args.class_name)
    if requested_classes:
        selected_classes = requested_classes
    elif class_names:
        selected_classes = class_names if len(class_names) <= 2 else [class_names[0]]
    elif dataset_type == "simplified":
        selected_classes = ["man"]
    else:
        selected_classes = ["armadillo"]

    meshes_per_label = int(args.meshes_per_label)
    if int(args.num_meshes) > 0:
        meshes_per_label = int(args.num_meshes)

    samples = _select_sample_meshes(
        dataset_type=dataset_type,
        dataset_root=dataset_root,
        class_names=selected_classes,
        mesh_path=args.mesh_path,
        meshes_per_label=meshes_per_label,
    )
    if not samples:
        raise RuntimeError("No sample meshes selected.")

    k_eig = int(args.k_eig)
    if k_eig <= 0:
        k_eig = int(ckpt_dataset.get("k_eig", 128))

    op_cache_dir = dataset_root.parent / "op_cache"
    op_cache_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    model = _build_diffusionnet_model(model_cfg)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    target_layer = _auto_target_layer(model, args.target_layer)

    viewer_map_name = args.viewer_map.strip().lower()
    viewer_quantity_name = f"{viewer_map_name}_density"
    viewer_meshes: list[ViewerMesh] = []
    maps_payload: dict[str, np.ndarray] = {}
    sample_reports: list[dict[str, Any]] = []
    all_passed = True
    first_saliency_density: np.ndarray | None = None
    first_ig_density: np.ndarray | None = None
    first_gradcam_density: np.ndarray | None = None

    for sample_index, sample in enumerate(samples):
        ops_model, ops_checks, verts_model, faces_model = _prepare_mesh_operators(
            vertices=sample.vertices,
            faces=sample.faces,
            k_eig=k_eig,
            op_cache_dir=op_cache_dir,
            device=device,
            diffusion_method=str(getattr(model, "diffusion_method", "spectral")),
        )

        if input_features == "xyz":
            features = verts_model
        else:
            assert isinstance(ops_model.evals, torch.Tensor)
            assert isinstance(ops_model.evecs, torch.Tensor)
            features = _build_hks_features(ops_model.evals, ops_model.evecs, count=16)

        forward_cfg = ForwardConfig(
            kwargs={"faces": faces_model},
            prefer_operator_signature=True,
        )

        with torch.no_grad():
            logits = run_model_forward(
                model,
                features,
                operators=ops_model,
                forward_config=forward_cfg,
            )

        if not isinstance(logits, torch.Tensor):
            raise TypeError("Expected tensor output from model forward.")

        logits_cpu = logits.detach().cpu().reshape(-1)
        pred_idx = int(torch.argmax(logits_cpu).item())
        pred_score = float(logits_cpu[pred_idx].item())
        pred_name = (
            class_names[pred_idx] if (0 <= pred_idx < len(class_names)) else f"class_{pred_idx}"
        )
        target_idx, target_name, target_source = _resolve_sample_target_label(
            sample_class_name=sample.class_name,
            class_names=class_names,
            n_classes=int(logits_cpu.numel()),
        )

        saliency = SaliencyExplainer(
            model=model,
            config=SaliencyConfig(
                aggregation="l2",
                mass_normalize=True,
                smooth="heat",
                smooth_tau=0.02,
                smooth_num_modes=64,
                prefer_operator_signature=True,
                forward_kwargs={"faces": faces_model},
            ),
        )
        saliency_result = saliency.explain(features, ops_model, target=target_idx)

        ig = IntegratedGradientsExplainer(
            model=model,
            config=IntegratedGradientsConfig(
                steps=int(args.ig_steps),
                baseline="heat",
                baseline_kwargs={"tau": 0.03},
                mass_normalize_gradients=True,
                smooth="heat",
                smooth_tau=0.01,
                prefer_operator_signature=True,
                forward_kwargs={"faces": faces_model},
            ),
        )
        ig_result = ig.explain(features, ops_model, target=target_idx)

        gradcam = IntrinsicGradCAMExplainer(
            model=model,
            config=GradCAMConfig(
                target_layer=target_layer,
                mass_weighted=True,
                signed=True,
                use_relu=False,
                smooth="heat",
                smooth_tau=0.02,
                prefer_operator_signature=True,
                forward_kwargs={"faces": faces_model},
            ),
        )
        gradcam_result = gradcam.explain(features, ops_model, target=target_idx)

        mass_check = check_mass_consistency(ops_checks)
        constant_check = check_constant_signal(ops_checks)
        ig_check = check_ig_completeness(
            ig_result.density_map.detach().cpu(),
            ops_checks,
            float(ig_result.metadata["completeness_rhs"]),
        )
        smooth_probe = saliency_result.smoothed_map
        if smooth_probe is None:
            smooth_probe = saliency_result.density_map
        smoothing_check = check_smoothing_energy(
            saliency_result.density_map.detach().cpu().numpy(),
            smooth_probe.detach().cpu().numpy(),
            ops_checks,
        )

        saliency_density = saliency_result.density_map.detach().cpu().numpy()
        ig_density = ig_result.density_map.detach().cpu().numpy()
        gradcam_density = gradcam_result.density_map.detach().cpu().numpy()
        if sample_index == 0:
            first_saliency_density = saliency_density
            first_ig_density = ig_density
            first_gradcam_density = gradcam_density

        sample_key = f"sample_{sample_index:02d}_{sample.class_name}"
        maps_payload[f"{sample_key}_saliency_density"] = saliency_density
        maps_payload[f"{sample_key}_ig_density"] = ig_density
        maps_payload[f"{sample_key}_gradcam_density"] = gradcam_density

        viewer_meshes.append(
            ViewerMesh(
                mesh_name=f"xai_mesh_{sample_index + 1}_{sample.class_name}",
                vertices=sample.vertices,
                faces=sample.faces,
                maps={
                    "ig": ig_result.density_map,
                    "saliency": saliency_result.density_map,
                    "gradcam": gradcam_result.density_map,
                },
                operators=ops_checks,
            )
        )

        gradcam_has_channels = "channel_weights" in gradcam_result.metadata
        sample_passed = all(
            [
                mass_check.passed,
                constant_check.passed,
                ig_check.passed,
                smoothing_check.passed,
                gradcam_has_channels,
            ]
        )
        all_passed = all_passed and sample_passed

        sample_reports.append(
            {
                "sample_index": sample_index,
                "sample": {
                    "class_name": sample.class_name,
                    "source_path": sample.source_path,
                    "num_vertices": int(sample.vertices.shape[0]),
                    "num_faces": int(sample.faces.shape[0]),
                },
                "predicted": {
                    "index": pred_idx,
                    "name": pred_name,
                    "score": pred_score,
                },
                "target": {
                    "index": target_idx,
                    "name": target_name,
                    "source": target_source,
                },
                "gradcam": {
                    "target_layer": target_layer,
                    "activation_shape": _jsonable(gradcam_result.metadata.get("activation_shape")),
                },
                "checks": {
                    mass_check.name: {"value": mass_check.value, "passed": mass_check.passed},
                    constant_check.name: {
                        "value": constant_check.value,
                        "passed": constant_check.passed,
                    },
                    ig_check.name: {"value": ig_check.value, "passed": ig_check.passed},
                    smoothing_check.name: {
                        "value": smoothing_check.value,
                        "passed": smoothing_check.passed,
                    },
                    "gradcam_channel_weights_present": gradcam_has_channels,
                },
                "metadata": {
                    "saliency": _jsonable(saliency_result.metadata),
                    "integrated_gradients": _jsonable(ig_result.metadata),
                    "gradcam": _jsonable(gradcam_result.metadata),
                },
                "shapes": {
                    "saliency": list(saliency_result.density_map.shape),
                    "ig": list(ig_result.density_map.shape),
                    "gradcam": list(gradcam_result.density_map.shape),
                },
                "passed": sample_passed,
            }
        )

    run_name = checkpoint_path.stem
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if len(sample_reports) == 1:
        if first_saliency_density is not None:
            maps_payload["saliency_density"] = first_saliency_density
        if first_ig_density is not None:
            maps_payload["ig_density"] = first_ig_density
        if first_gradcam_density is not None:
            maps_payload["gradcam_density"] = first_gradcam_density
    np.savez(run_dir / "maps.npz", **maps_payload)

    viewer_status = "SKIPPED"
    viewer_error = ""
    try:
        if args.show_polyscope:
            _launch_polyscope_method_picker(
                viewer_meshes=viewer_meshes,
                initial_map=viewer_map_name,
                initial_method=args.viewer_method,
                initial_tau=float(args.viewer_tau),
                tau_min=0.0,
                tau_max=float(args.viewer_tau_max),
            )
            viewer_status = "OPENED"
        else:
            first_entry = viewer_meshes[0]
            if viewer_map_name == "ig":
                viewer_map = first_entry.maps["ig"]
            elif viewer_map_name == "saliency":
                viewer_map = first_entry.maps["saliency"]
            else:
                viewer_map = first_entry.maps["gradcam"]
            launch_polyscope_viewer(
                vertices=first_entry.vertices,
                faces=first_entry.faces,
                scalar_map=viewer_map.detach().cpu(),
                operators=first_entry.operators,
                config=PolyscopeViewerConfig(
                    mesh_name="xai_mesh",
                    quantity_name=viewer_quantity_name,
                    collapse="first",
                    smoothing_method=args.viewer_method,
                    initial_tau=float(args.viewer_tau),
                    tau_min=0.0,
                    tau_max=float(args.viewer_tau_max),
                    show=False,
                ),
            )
            viewer_status = "READY"
    except RuntimeError as exc:
        if args.show_polyscope:
            raise
        viewer_status = "UNAVAILABLE"
        viewer_error = str(exc)

    report = {
        "checkpoint": str(checkpoint_path),
        "dataset_type": dataset_type,
        "dataset_root": str(dataset_root),
        "labels_selected": selected_classes,
        "meshes_per_label": meshes_per_label,
        "samples": sample_reports,
        "device": str(device),
        "k_eig": int(k_eig),
        "input_features": input_features,
        "viewer": {
            "map": viewer_map_name,
            "status": viewer_status,
            "error": viewer_error,
        },
        "overall_passed": all_passed,
    }

    with (run_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print("== GeoInspect x DiffusionNet sanity check ==")
    print("checkpoint:", checkpoint_path)
    print("samples selected:", len(sample_reports))
    for sample_report in sample_reports:
        sample_info = sample_report["sample"]
        predicted = sample_report["predicted"]
        target = sample_report["target"]
        shapes = sample_report["shapes"]
        print(
            f"[sample {sample_report['sample_index']}]",
            sample_info["class_name"],
            sample_info["source_path"],
        )
        print(
            "  prediction:",
            predicted["index"],
            predicted["name"],
            f"score={predicted['score']:.6f}",
        )
        print("  target:", target["index"], target["name"], f"source={target['source']}")
        print(
            "  shapes:",
            f"saliency={tuple(shapes['saliency'])}",
            f"ig={tuple(shapes['ig'])}",
            f"gradcam={tuple(shapes['gradcam'])}",
        )
        checks = sample_report["checks"]
        print(
            "  checks:",
            f"mass_consistency_total_area={checks['mass_consistency_total_area']['passed']}",
            f"constant_signal_response={checks['constant_signal_response']['passed']}",
            f"ig_completeness_error={checks['ig_completeness_error']['passed']}",
            f"smoothing_energy_ratio={checks['smoothing_energy_ratio']['passed']}",
            f"gradcam_channel_weights_present={checks['gradcam_channel_weights_present']}",
        )
    print("gradcam target layer:", target_layer)
    print("viewer map:", viewer_map_name)
    print("artifacts:", run_dir)
    if viewer_status == "UNAVAILABLE":
        print("POLYSCOPE: UNAVAILABLE (install polyscope to enable GUI checks)")
    elif viewer_status == "OPENED":
        print("POLYSCOPE: OPENED")
    else:
        print("POLYSCOPE: READY (run with --show_polyscope to open GUI)")
    print("STATUS:", "PASS" if all_passed else "FAIL")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
