"""
tests/test_main_pipeline.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
End-to-end integration test for main.py pipeline orchestrator.
Tests the upgraded multi-tile GCP aggregation path.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from main import run_pipeline, load_config
from scripts.generate_synthetic_geotiff import create_synthetic_geotiffs


@pytest.fixture
def synthetic_data(tmp_path: Path) -> tuple[Path, Path]:
    return create_synthetic_geotiffs(
        output_dir=tmp_path / "data",
        width=512,
        height=512,
        shift_x=10.0,
        shift_y=-5.0,
        angle_deg=1.0,
    )


def test_create_synthetic_geotiffs_default_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    ref_path, tgt_path = create_synthetic_geotiffs(width=128, height=128)

    assert ref_path.resolve() == (tmp_path / "data" / "reference.tif").resolve()
    assert tgt_path.resolve() == (tmp_path / "data" / "target.tif").resolve()
    assert ref_path.exists()
    assert tgt_path.exists()


def test_load_config_valid(tmp_path: Path) -> None:
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("keypoints:\n  method: SIFT\n", encoding="utf-8")
    assert load_config(cfg_file)["keypoints"]["method"] == "SIFT"


def test_load_config_missing_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("no_such_file.yaml")


def test_run_pipeline_end_to_end(synthetic_data: tuple[Path, Path], tmp_path: Path) -> None:
    ref_path, tgt_path = synthetic_data
    out_path = tmp_path / "output" / "registered.tif"

    config = {
        "tiling":     {"tile_size": 256, "overlap_pct": 0.1},
        "keypoints":  {"method": "SIFT", "max_keypoints": 1000, "grid_cells": 4},
        "flann":      {"trees": 5, "checks": 50, "ratio_threshold": 0.75},
        "ransac":     {"threshold": 5.0, "max_iter": 1000, "confidence": 0.99, "model": "homography"},
        "geospatial": {"crs_target": "EPSG:4326"},
        "evaluation": {"min_inliers": 4, "max_rmse": 15.0, "min_coverage": 0.01,
                       "min_tile_success_rate": 0.1},
    }

    summary = run_pipeline(ref_path, tgt_path, out_path, config)

    assert out_path.exists(), "Output GeoTIFF must be created"
    assert summary["evaluation"]["overall_pass"] is True, f"Pipeline failed: {summary['warnings']}"
    assert summary["metrics"]["global_rmse_px"] is not None
    assert summary["metrics"]["global_rmse_px"] < 5.0

    summary_file = out_path.parent / "registered_summary.json"
    assert summary_file.exists()
    with open(summary_file, encoding="utf-8") as f:
        d = json.load(f)
    assert d["status"] == "SUCCESS"
