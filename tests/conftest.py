"""Shared pytest fixtures for GeoInspect."""

from __future__ import annotations

import pytest

from geoinspect import MeshOperators


@pytest.fixture()
def empty_ops() -> MeshOperators:
    """Return a minimal operators container for contract tests."""
    return MeshOperators(mass=None)
