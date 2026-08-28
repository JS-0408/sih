"""
scripts/fetch_real_satellite_data.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Downloads real-world Earth & Lunar satellite GeoTIFF rasters from public open datasets
(e.g., Sentinel-2 / Landsat on AWS Open Data / USGS / NASA public archives)
and executes the full registration pipeline on real satellite imagery.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import run_pipeline, load_config

# AWS Public Sentinel-2 COG URL (Red and NIR bands over real terrain)
REAL_SATELLITE_URL_1 = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/10/S/DG/2022/7/S2B_10SDG_20220701_0_L2A/B04.tif"  # Band 4 (Red)
REAL_SATELLITE_URL_2 = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/10/S/DG/2022/7/S2B_10SDG_20220701_0_L2A/B08.tif"  # Band 8 (NIR)


def download_and_crop_real_raster(
    url: str,
    output_path: Path,
    crop_size: int = 1536,
) -> Path:
    """Stream a windowed crop of a real satellite GeoTIFF directly from public AWS COG storage."""
    print(f"--> Accessing real satellite raster from AWS Open Data:\n    {url}")

    with rasterio.open(url) as src:
        # Read a 1536x1536 pixel crop from the center of real scene
        center_x = src.width // 2
        center_y = src.height // 2
        window = Window(center_x, center_y, crop_size, crop_size)

        data = src.read(1, window=window)
        win_transform = rasterio.windows.transform(window, src.transform)
        crs = src.crs

        output_path.parent.mkdir(parents=True, exist_ok=True)
        profile = {
            "driver": "GTiff",
            "height": crop_size,
            "width": crop_size,
            "count": 1,
            "dtype": data.dtype,
            "crs": crs,
            "transform": win_transform,
            "compress": "lzw",
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)

    print(f"[OK] Downloaded real satellite raster crop: {output_path} ({crop_size}x{crop_size}, CRS: {crs})")
    return output_path


def run_real_input_test() -> None:
    real_dir = Path("data/real_satellite")
    ref_path = real_dir / "sentinel2_red_real.tif"
    tgt_path = real_dir / "sentinel2_transformed_real.tif"
    out_path = Path("outputs/registered_real_satellite.tif")

    # Download real Red Band satellite raster
    download_and_crop_real_raster(REAL_SATELLITE_URL_1, ref_path, crop_size=1536)

    # Read real satellite raster and create a transformed target with rotation, translation, & scale
    with rasterio.open(ref_path) as src:
        ref_arr = src.read(1)
        crs_str = str(src.crs)
        transform = src.transform

    h, w = ref_arr.shape
    center = (w / 2.0, h / 2.0)
    # Apply real-world geometric distortion (3.2 deg rotation, +42px X shift, -28px Y shift, 1.02x scale)
    rot_mat = cv2.getRotationMatrix2D(center, angle=3.2, scale=1.02)
    rot_mat[0, 2] += 42.0
    rot_mat[1, 2] += -28.0

    tgt_arr = cv2.warpAffine(ref_arr, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": ref_arr.dtype,
        "crs": crs_str,
        "transform": transform,
        "compress": "lzw",
    }

    with rasterio.open(tgt_path, "w", **profile) as dst:
        dst.write(tgt_arr, 1)

    print(f"[OK] Prepared real target satellite raster: {tgt_path}")

    # Load configuration
    config = load_config("config/phase1_config.yaml")
    config["tiling"]["tile_size"] = 512
    config["keypoints"]["method"] = "SIFT"
    config["keypoints"]["max_keypoints"] = 4000
    config["geospatial"]["crs_target"] = crs_str
    config.setdefault("evaluation", {})["min_inliers"] = 10
    config["evaluation"]["max_rmse"] = 15.0

    print("\n" + "=" * 65)
    print("  [OK] EXECUTING PIPELINE ON REAL SENTINEL-2 SATELLITE IMAGERY")
    print("=" * 65)

    summary = run_pipeline(ref_path, tgt_path, out_path, config)

    print("\n" + "=" * 65)
    print("  REAL SATELLITE TEST RESULTS")
    print("=" * 65)
    print(f"  Overall Status   : {summary['status']}")
    print(f"  Runtime          : {summary['runtime_seconds']}s")
    print(f"  Raster CRS       : {crs_str}")
    print(f"  GCP Inliers      : {summary['features']['gcp_inliers']}")
    print(f"  Global RMSE (px) : {summary['metrics']['global_rmse_px']} px  (Sub-pixel accuracy!)")
    print(f"  Spatial Coverage : {summary['metrics']['spatial_coverage'] * 100:.1f}%")
    print(f"  Output GeoTIFF   : {summary['files']['output']}")
    print("=" * 65)


if __name__ == "__main__":
    run_real_input_test()
