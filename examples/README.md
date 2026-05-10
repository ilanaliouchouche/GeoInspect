# Examples

These scripts show how to use GeoInspect explainers, checks, and visualization.

## Prerequisites

Run commands from the repository root.

Minimal setup:

```bash
pip install -e .
```

Optional extras (recommended for all examples):

```bash
pip install -e ".[viz,diffusionnet]"
```

## Quick Run Guide

| Script | Command | What it does |
|---|---|---|
| `01_mass_saliency.py` | `python examples/01_mass_saliency.py` | Runs mass-normalized saliency on a tiny toy model and prints raw/density/contribution maps. |
| `02_integrated_gradients_heat_baseline.py` | `python examples/02_integrated_gradients_heat_baseline.py` | Runs Integrated Gradients with a heat baseline and prints completeness error. |
| `03_integrated_gradients_spectral_baseline.py` | `python examples/03_integrated_gradients_spectral_baseline.py` | Runs Integrated Gradients with a spectral low-pass baseline and prints completeness error. |
| `04_intrinsic_gradcam.py` | `python examples/04_intrinsic_gradcam.py` | Runs intrinsic Grad-CAM on a toy model and prints maps and channel weights. |
| `05_smoothing_comparison.py` | `python examples/05_smoothing_comparison.py` | Generates IG artifacts in `outputs/example05/` and opens Polyscope with an interactive `t (tau)` smoothing slider. |
| `06_diffusionnet_readiness_check.py` | `python examples/06_diffusionnet_readiness_check.py` | Runs a reproducible readiness certification (saliency + IG + Grad-CAM + mathematical checks) and prints `PASS`/`FAIL`. |
| `07_diffusionnet_xai_sanity.py` | `python examples/07_diffusionnet_xai_sanity.py ...` | End-to-end XAI sanity run on a trained DiffusionNet checkpoint, with optional interactive Polyscope UI. |

## End-to-End Example (07)

```bash
python examples/07_diffusionnet_xai_sanity.py \
  --checkpoint <path/to/diffusion_net.pt> \
  --dataset_type original (or simplifie)\
  --dataset_root <path/to/shrec11/data/original_or_simplified> \
  --input_features hks \
  --device auto \
  --class_name man,woman \
  --meshes_per_label 2 \
  --show_polyscope
```

This command:
- Loads a trained DiffusionNet checkpoint.
- Selects meshes per class (`--meshes_per_label`).
- Runs Saliency, Integrated Gradients, and intrinsic Grad-CAM.
- Runs consistency checks and saves `report.json` + `maps.npz`.
- Opens Polyscope with method/map controls and an interactive `t (tau)` slider.

## Notes

- If Polyscope is not installed, GUI examples will fail until `viz` dependencies are installed.
- Script `07` requires a valid DiffusionNet checkpoint and the matching SHREC11 dataset layout.
