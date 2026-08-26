"""Tests for src/io/raster_loader.py"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from src.io.raster_loader import InvalidCRSError, RasterLoader, Tile


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _write_geotiff(path: Path, crs: str = "EPSG:4326", bands: int = 3, size: int = 64) -> None:
    """Create a minimal synthetic GeoTIFF for testing."""
    transform = from_bounds(0, 0, 1, 1, size, size)
    data = np.random.randint(0, 255, (bands, size, size), dtype=np.uint8)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        width=size,
        height=size,
        count=bands,
        dtype="uint8",
        crs=CRS.from_string(crs),
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture
def valid_tiff(tmp_path: Path) -> Path:
    p = tmp_path / "valid.tif"
    _write_geotiff(p)
    return p


@pytest.fixture
def wrong_crs_tiff(tmp_path: Path) -> Path:
    p = tmp_path / "wrong_crs.tif"
    _write_geotiff(p, crs="EPSG:32643")
    return p


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestRasterLoaderInit:
    def test_default_init(self) -> None:
        loader = RasterLoader()
        assert loader.crs_target == CRS.from_string("EPSG:4326")
        assert loader.bands is None
        assert loader.max_memory_mb == 4096

    def test_custom_init(self) -> None:
        loader = RasterLoader(crs_target="EPSG:4326", bands=[1], max_memory_mb=512)
        assert loader.bands == [1]
        assert loader.max_memory_mb == 512


class TestRasterLoaderLoad:
    def test_load_valid_tiff(self, valid_tiff: Path) -> None:
        loader = RasterLoader()
        data, meta = loader.load(valid_tiff)
        assert data.ndim == 3
        assert data.shape[0] == 3       # 3 bands
        assert data.shape[1:] == (64, 64)
        assert meta.crs == CRS.from_string("EPSG:4326")

    def test_load_missing_file(self) -> None:
        loader = RasterLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.tif")

    def test_load_wrong_crs(self, wrong_crs_tiff: Path) -> None:
        loader = RasterLoader(crs_target="EPSG:4326")
        with pytest.raises(InvalidCRSError):
            loader.load(wrong_crs_tiff)

    def test_load_specific_bands(self, valid_tiff: Path) -> None:
        loader = RasterLoader(bands=[1, 2])
        data, meta = loader.load(valid_tiff)
        assert data.shape[0] == 2
        assert meta.count == 2


class TestTileGenerator:
    def test_tiles_are_correct_size(self, valid_tiff: Path) -> None:
        loader = RasterLoader()
        data, meta = loader.load(valid_tiff)
        tiles = list(loader.tile_generator(data, meta, tile_size=32, overlap_pct=0.0))
        assert len(tiles) > 0
        for tile in tiles:
            assert isinstance(tile, Tile)
            assert tile.data.shape[1:] == (32, 32)

    def test_tiles_have_transforms(self, valid_tiff: Path) -> None:
        loader = RasterLoader()
        data, meta = loader.load(valid_tiff)
        tiles = list(loader.tile_generator(data, meta, tile_size=32, overlap_pct=0.0))
        for tile in tiles:
            assert tile.transform is not None

    def test_invalid_overlap_raises(self, valid_tiff: Path) -> None:
        loader = RasterLoader()
        data, meta = loader.load(valid_tiff)
        with pytest.raises(ValueError):
            list(loader.tile_generator(data, meta, tile_size=32, overlap_pct=1.0))
