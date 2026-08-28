from __future__ import annotations

from pathlib import Path

import numpy as np

import server


def test_preview_rejects_outside_allowed_roots(tmp_path: Path, monkeypatch) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside_file = tmp_path / "outside.tif"
    outside_file.write_bytes(b"x")
    monkeypatch.setattr(server, "ALLOWED_PATH_ROOTS", (allowed_root.resolve(),))

    client = server.app.test_client()
    resp = client.get(f"/api/preview?path={outside_file}")
    assert resp.status_code == 403


def test_preview_allows_files_inside_allowed_roots(tmp_path: Path, monkeypatch) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    inside_file = allowed_root / "inside.tif"
    inside_file.write_bytes(b"x")
    monkeypatch.setattr(server, "ALLOWED_PATH_ROOTS", (allowed_root.resolve(),))

    class _FakeLoader:
        def load(self, _path):
            return np.zeros((8, 8), dtype=np.uint8), None

    monkeypatch.setattr(server, "RasterLoader", _FakeLoader)

    client = server.app.test_client()
    resp = client.get(f"/api/preview?path={inside_file}")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"


def test_outputs_route_rejects_path_traversal(tmp_path: Path, monkeypatch) -> None:
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    monkeypatch.setattr(server, "OUTPUTS_DIR", outputs_dir.resolve())

    client = server.app.test_client()
    resp = client.get("/outputs/../secret.txt")
    assert resp.status_code == 403
