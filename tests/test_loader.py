"""
tests/test_loader.py
~~~~~~~~~~~~~~~~~~~~
Copyright (c) 2026 Santhosh Jayakumar & Team — MIT License

Unit tests for windowed raster loader & memory safety.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from src.io.raster_loader import RasterLoader, Tile, RasterMeta


def test_raster_loader_load(tmp_path: Path):
    import rasterio
    from rasterio.transform import from_origin

    tiff_path = tmp_path / "test.tif"
    profile = {
        "driver": "GTiff",
        "height": 256,
        "width": 256,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_origin(77.0, 12.0, 0.0001, 0.0001),
    }
    data = np.full((256, 256), 100, dtype=np.uint8)
    with rasterio.open(tiff_path, "w", **profile) as dst:
        dst.write(data, 1)

    loader = RasterLoader()
    arr, meta = loader.load(tiff_path)
    assert arr.shape == (1, 256, 256)
    assert meta.crs.to_string() == "EPSG:4326"


def test_windowed_tile_generator(tmp_path: Path):
    import rasterio
    from rasterio.transform import from_origin

    tiff_path = tmp_path / "test.tif"
    profile = {
        "driver": "GTiff",
        "height": 512,
        "width": 512,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_origin(77.0, 12.0, 0.0001, 0.0001),
    }
    data = np.full((512, 512), 150, dtype=np.uint8)
    with rasterio.open(tiff_path, "w", **profile) as dst:
        dst.write(data, 1)

    loader = RasterLoader()
    tiles = list(loader.tile_generator(tiff_path, tile_size=256, overlap_pct=0.0))
    assert len(tiles) == 4
    for tile in tiles:
        assert isinstance(tile, Tile)
        assert tile.data.shape == (1, 256, 256)
