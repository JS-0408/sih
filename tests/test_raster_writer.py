"""Tests for src/io/raster_writer.py"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from src.io.raster_writer import RasterWriter


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_array(bands: int = 3, h: int = 64, w: int = 64, dtype=np.uint8) -> np.ndarray:
    """Return a synthetic (bands, H, W) array."""
    rng = np.random.default_rng(42)
    if np.issubdtype(dtype, np.floating):
        return rng.random((bands, h, w)).astype(dtype)
    return rng.integers(0, 255, (bands, h, w), dtype=dtype)


def _default_transform() -> rasterio.transform.Affine:
    return from_bounds(0, 0, 1, 1, 64, 64)


# ─── Init tests ───────────────────────────────────────────────────────────────

class TestRasterWriterInit:
    def test_default_init(self) -> None:
        writer = RasterWriter()
        assert writer.crs == CRS.from_string("EPSG:4326")
        assert writer.compress == "lzw"
        assert writer.nodata is None

    def test_custom_init(self) -> None:
        writer = RasterWriter(crs="EPSG:4326", compress="deflate", nodata=0.0)
        assert writer.compress == "deflate"
        assert writer.nodata == 0.0


# ─── Write tests ──────────────────────────────────────────────────────────────

class TestRasterWriterWrite:
    def test_writes_valid_geotiff(self, tmp_path: Path) -> None:
        """Written file must be openable by rasterio."""
        writer = RasterWriter()
        array = _make_array(dtype=np.uint8)
        transform = _default_transform()
        out = writer.write(array, transform, tmp_path / "out.tif")
        assert out.exists()
        with rasterio.open(out) as src:
            assert src.count == 3

    def test_crs_preserved(self, tmp_path: Path) -> None:
        """CRS embedded in output must match writer configuration."""
        writer = RasterWriter(crs="EPSG:4326")
        out = writer.write(_make_array(), _default_transform(), tmp_path / "crs.tif")
        with rasterio.open(out) as src:
            assert src.crs == CRS.from_string("EPSG:4326")

    def test_transform_preserved(self, tmp_path: Path) -> None:
        """Affine transform must round-trip through the file."""
        transform = _default_transform()
        writer = RasterWriter()
        out = writer.write(_make_array(), transform, tmp_path / "tfm.tif")
        with rasterio.open(out) as src:
            # Rasterio may re-read with minor float precision diff; check c, e (origin)
            assert abs(src.transform.c - transform.c) < 1e-9
            assert abs(src.transform.f - transform.f) < 1e-9

    def test_pixel_data_preserved(self, tmp_path: Path) -> None:
        """Pixel values written must match values read back."""
        array = _make_array(dtype=np.uint8)
        writer = RasterWriter()
        out = writer.write(array, _default_transform(), tmp_path / "data.tif")
        with rasterio.open(out) as src:
            data_back = src.read()
        np.testing.assert_array_equal(data_back, array)

    def test_uint16_dtype(self, tmp_path: Path) -> None:
        """uint16 arrays must write and read back correctly."""
        array = _make_array(dtype=np.uint16)
        writer = RasterWriter()
        out = writer.write(array, _default_transform(), tmp_path / "u16.tif")
        with rasterio.open(out) as src:
            assert src.dtypes[0] == "uint16"

    def test_float32_dtype(self, tmp_path: Path) -> None:
        """float32 arrays must write and read back correctly."""
        array = _make_array(dtype=np.float32)
        writer = RasterWriter()
        out = writer.write(array, _default_transform(), tmp_path / "f32.tif")
        with rasterio.open(out) as src:
            assert src.dtypes[0] == "float32"

    def test_single_band_2d_array(self, tmp_path: Path) -> None:
        """A 2-D (H, W) array must be promoted to (1, H, W) and written."""
        array_2d = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        writer = RasterWriter()
        out = writer.write(array_2d, _default_transform(), tmp_path / "gray.tif")
        with rasterio.open(out) as src:
            assert src.count == 1

    def test_overwrite_false_raises(self, tmp_path: Path) -> None:
        """Writing to an existing path with overwrite=False must raise."""
        writer = RasterWriter()
        out_path = tmp_path / "exist.tif"
        writer.write(_make_array(), _default_transform(), out_path)
        with pytest.raises(FileExistsError):
            writer.write(_make_array(), _default_transform(), out_path, overwrite=False)

    def test_overwrite_true_replaces_file(self, tmp_path: Path) -> None:
        """overwrite=True must silently replace an existing file."""
        writer = RasterWriter()
        out_path = tmp_path / "replace.tif"
        writer.write(_make_array(), _default_transform(), out_path)
        writer.write(_make_array(), _default_transform(), out_path, overwrite=True)
        assert out_path.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Non-existent parent directories must be created automatically."""
        writer = RasterWriter()
        deep_path = tmp_path / "a" / "b" / "c" / "out.tif"
        writer.write(_make_array(), _default_transform(), deep_path)
        assert deep_path.exists()

    def test_nodata_embedded(self, tmp_path: Path) -> None:
        """nodata value must appear in the written file metadata."""
        writer = RasterWriter(nodata=0.0)
        out = writer.write(_make_array(), _default_transform(), tmp_path / "nd.tif")
        with rasterio.open(out) as src:
            assert src.nodata == 0.0


# ─── write_tile convenience method ────────────────────────────────────────────

class TestRasterWriterWriteTile:
    def test_tile_filename_format(self, tmp_path: Path) -> None:
        """write_tile() must create a file named tile_<tile_id>.tif."""
        writer = RasterWriter()
        out = writer.write_tile(
            _make_array(), _default_transform(), tmp_path, tile_id="r0_c1"
        )
        assert out.name == "tile_r0_c1.tif"
        assert out.exists()

    def test_tile_data_readable(self, tmp_path: Path) -> None:
        """Tile file written via write_tile() must be openable by rasterio."""
        writer = RasterWriter()
        out = writer.write_tile(
            _make_array(), _default_transform(), tmp_path, tile_id="r1_c0"
        )
        with rasterio.open(out) as src:
            assert src.count == 3
