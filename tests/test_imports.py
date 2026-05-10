"""Import and version smoke tests."""

from __future__ import annotations

import geoinspect


def test_version_is_exposed() -> None:
    assert isinstance(geoinspect.__version__, str)
    assert geoinspect.__version__
