from __future__ import annotations

from pathlib import Path

import rasterio
from rasterio.io import MemoryFile


def validate_geotiff_upload(
    filename: str,
    payload: bytes,
    max_bytes: int = 200 * 1024 * 1024,
) -> bytes:
    if not filename:
        raise ValueError("Uploaded file must have a filename.")

    suffix = Path(filename).suffix.lower()
    if suffix not in {".tif", ".tiff"}:
        raise ValueError("Only .tif/.tiff files are allowed.")

    if not payload:
        raise ValueError("Uploaded file is empty.")

    if len(payload) > max_bytes:
        raise ValueError(f"Uploaded file exceeds size limit of {max_bytes} bytes.")

    try:
        with MemoryFile(payload) as memfile:
            with memfile.open() as ds:
                if ds.driver != "GTiff":
                    raise ValueError("Only GeoTIFF uploads are supported.")
                if ds.width <= 0 or ds.height <= 0 or ds.count <= 0:
                    raise ValueError("Invalid GeoTIFF dimensions or band count.")
    except rasterio.errors.RasterioError as exc:
        raise ValueError(f"Invalid GeoTIFF upload: {exc}") from exc

    return payload
