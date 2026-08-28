from __future__ import annotations

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from src.io.upload_validation import validate_geotiff_upload


def _make_tiff_bytes() -> bytes:
    arr = np.arange(100, dtype=np.uint8).reshape(10, 10)
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=arr.dtype,
            transform=from_origin(0, 0, 1, 1),
        ) as ds:
            ds.write(arr, 1)
        return memfile.read()


def test_validate_geotiff_upload_accepts_valid_tiff() -> None:
    payload = _make_tiff_bytes()
    out = validate_geotiff_upload("input.tif", payload, max_bytes=10_000_000)
    assert out == payload


def test_validate_geotiff_upload_rejects_non_tiff_extension() -> None:
    with pytest.raises(ValueError, match="Only .tif/.tiff"):
        validate_geotiff_upload("input.png", b"not-a-tiff")


def test_validate_geotiff_upload_rejects_oversized_payload() -> None:
    payload = _make_tiff_bytes()
    with pytest.raises(ValueError, match="size limit"):
        validate_geotiff_upload("input.tif", payload, max_bytes=10)


def test_validate_geotiff_upload_rejects_invalid_tiff_bytes() -> None:
    with pytest.raises(ValueError, match="Invalid GeoTIFF"):
        validate_geotiff_upload("input.tif", b"bad-data")
