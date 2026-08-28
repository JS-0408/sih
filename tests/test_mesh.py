"""
tests/test_mesh.py
~~~~~~~~~~~~~~~~~~~
Copyright (c) 2026 Santhosh Jayakumar & Team — MIT License

Unit tests for src/geometry/mesh_builder.SurfaceMeshBuilder.
Tests cover: synthetic DEM generation, HTML export (fallback path),
Z-scale and downsample parameter handling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def builder():
    from src.geometry.mesh_builder import SurfaceMeshBuilder
    return SurfaceMeshBuilder(z_scale=2.0, downsample=4)


@pytest.fixture
def synthetic_dem(builder):
    return builder.generate_synthetic_dem(width=128, height=128, seed=42)


@pytest.fixture
def tmp_html(tmp_path: Path) -> Path:
    return tmp_path / "terrain_3d.html"


# ─────────────────────────────────────────────────────────────────────────────
# Tests — SurfaceMeshBuilder
# ─────────────────────────────────────────────────────────────────────────────

class TestSurfaceMeshBuilderInit:
    def test_default_init(self):
        from src.geometry.mesh_builder import SurfaceMeshBuilder
        b = SurfaceMeshBuilder()
        assert b.z_scale == 1.5
        assert b.downsample == 4

    def test_custom_init(self):
        from src.geometry.mesh_builder import SurfaceMeshBuilder
        b = SurfaceMeshBuilder(z_scale=5.0, downsample=8)
        assert b.z_scale == 5.0
        assert b.downsample == 8

    def test_downsample_minimum_clamped(self):
        from src.geometry.mesh_builder import SurfaceMeshBuilder
        b = SurfaceMeshBuilder(downsample=0)
        assert b.downsample == 1


class TestSyntheticDEM:
    def test_shape_correct(self, builder):
        dem = builder.generate_synthetic_dem(width=64, height=48)
        assert dem.shape == (48, 64)

    def test_dtype_float32(self, builder):
        dem = builder.generate_synthetic_dem()
        assert dem.dtype == np.float32

    def test_contains_craters(self, builder):
        """DEM should have non-trivial range due to added crater features."""
        dem = builder.generate_synthetic_dem(width=128, height=128)
        assert dem.max() - dem.min() > 0.05

    def test_reproducible_with_seed(self, builder):
        dem_a = builder.generate_synthetic_dem(seed=7)
        dem_b = builder.generate_synthetic_dem(seed=7)
        np.testing.assert_array_equal(dem_a, dem_b)

    def test_different_seeds_differ(self, builder):
        dem_a = builder.generate_synthetic_dem(seed=1)
        dem_b = builder.generate_synthetic_dem(seed=2)
        assert not np.array_equal(dem_a, dem_b)


class TestHTMLExport:
    def test_fallback_html_created(self, builder, synthetic_dem, tmp_html):
        """Fallback HTML export must always succeed even without PyVista."""
        out = builder._export_fallback_html(synthetic_dem, tmp_html)
        assert out.exists()
        assert out.suffix == ".html"

    def test_fallback_html_not_empty(self, builder, synthetic_dem, tmp_html):
        out = builder._export_fallback_html(synthetic_dem, tmp_html)
        content = out.read_text(encoding="utf-8")
        assert len(content) > 500

    def test_fallback_html_contains_elevation_keyword(self, builder, synthetic_dem, tmp_html):
        out = builder._export_fallback_html(synthetic_dem, tmp_html)
        content = out.read_text(encoding="utf-8")
        assert "Elevation" in content or "elevation" in content.lower()

    def test_build_and_export_creates_file(self, builder, synthetic_dem, tmp_html):
        """build_and_export must return a valid Path that exists."""
        out = builder.build_and_export(elevation=synthetic_dem, output_path=tmp_html)
        assert isinstance(out, Path)
        assert out.exists()

    def test_build_and_export_with_grayscale_texture(self, builder, synthetic_dem, tmp_html):
        """Build with a grayscale texture overlay must succeed."""
        texture = (synthetic_dem * 255).astype(np.uint8)
        out = builder.build_and_export(
            elevation=synthetic_dem,
            output_path=tmp_html,
            texture_image=texture,
        )
        assert out.exists()

    def test_build_and_export_creates_parent_dirs(self, builder, synthetic_dem, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "terrain.html"
        out = builder.build_and_export(elevation=synthetic_dem, output_path=deep_path)
        assert out.exists()


class TestFLANNRansacMatcher:
    """Unit tests for the consolidated FLANN+RANSAC matcher."""

    def test_import(self):
        from src.matching.flann_ransac import FLANNRansacMatcher
        m = FLANNRansacMatcher()
        assert m.ratio == 0.75

    def test_match_returns_tuple_of_three(self):
        from src.matching.flann_ransac import FLANNRansacMatcher
        import cv2
        m = FLANNRansacMatcher(max_keypoints=500)
        rng = np.random.default_rng(0)
        img = (rng.uniform(0, 255, (256, 256))).astype(np.uint8)
        src, dst, H = m.match(img, img)
        assert isinstance(src, np.ndarray)
        assert isinstance(dst, np.ndarray)
        assert src.shape[1] == 2 or src.shape[0] == 0

    def test_empty_image_returns_empty(self):
        from src.matching.flann_ransac import FLANNRansacMatcher
        m = FLANNRansacMatcher()
        blank = np.zeros((64, 64), dtype=np.uint8)
        src, dst, H = m.match(blank, blank)
        assert src.shape[0] == 0
        assert H is None
