from __future__ import annotations

from pathlib import Path

import pytest

from src.io.path_policy import resolve_allowed_path


def test_resolve_allowed_path_accepts_path_inside_root(tmp_path: Path) -> None:
    allowed_root = tmp_path / "data"
    allowed_root.mkdir()
    f = allowed_root / "sample.tif"
    f.write_bytes(b"demo")

    resolved = resolve_allowed_path(f, tmp_path, [allowed_root])
    assert resolved == f.resolve()


def test_resolve_allowed_path_rejects_relative_traversal(tmp_path: Path) -> None:
    allowed_root = tmp_path / "data"
    allowed_root.mkdir()

    with pytest.raises(PermissionError):
        resolve_allowed_path("../outside.txt", allowed_root, [allowed_root])


def test_resolve_allowed_path_rejects_absolute_outside_root(tmp_path: Path) -> None:
    allowed_root = tmp_path / "data"
    allowed_root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")

    with pytest.raises(PermissionError):
        resolve_allowed_path(outside, tmp_path, [allowed_root])
