from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="geoinspect",
    version="0.1.0",
    description="Mesh-aware explainability toolkit for DiffusionNet on triangular meshes.",
    long_description=README,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"geoinspect": ["py.typed"]},
    include_package_data=True,
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        "diffusionnet": [
            "torch>=2.2",
            "scipy>=1.11",
            "scikit-learn>=1.4",
            "potpourri3d>=1.2.1",
            "robust-laplacian>=0.2.7",
            "tqdm>=4.66",
            "plyfile>=1.0",
        ],
        "diffusionnet-geodesic": [
            "libigl>=2.5",
        ],
        "geometry": [
            "scipy>=1.11",
        ],
        "xai": [
            "captum>=0.7",
        ],
        "viz": [
            "polyscope>=2.3",
            "matplotlib>=3.8",
        ],
        "test": [
            "pytest>=8.3",
            "pytest-cov>=5.0",
        ],
        "lint": [
            "ruff>=0.8.0",
            "mypy>=1.13",
        ],
    },
)
