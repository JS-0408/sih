"""
scripts/test_uint16_orb_scene.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Test pipeline with 16-bit unsigned integer (uint16) single-band raster
using ORB feature detector backend and EPSG:3857 (Web Mercator) CRS projection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import run_pipeline, load_config


def run_uint16_orb_test() -> None:
    out_dir = Path("data/uint16_scene")
    out_dir.mkdir(parents=True, exist_ok=True)

    width, height = 1024, 1024
    ref_path = out_dir / "reference_uint16.tif"
    tgt_path = out_dir / "target_uint16.tif"
    output_path = Path("outputs/registered_uint16_orb.tif")

    # Generate synthetic 16-bit image array (0 to 65535 values)
    np.random.seed(42)
    base = np.random.normal(30000, 5000, (height, width))

    for _ in range(30):
        cx, cy = np.random.randint(100, 900, 2)
        r = np.random.randint(30, 80)
        cv2.circle(base, (cx, cy), r, (50000,), 6)
        cv2.circle(base, (cx + 5, cy + 5), r - 10, (10000,), -1)

    ref_arr = np.clip(base, 0, 65535).astype(np.uint16)

    # Apply rigid shift to target
    M = np.float32([[1, 0, 15.0], [0, 1, -12.0]])
    tgt_arr = cv2.warpAffine(ref_arr, M, (width, height))

    crs_str = "EPSG:3857"
    transform = from_origin(500000.0, 4000000.0, 2.0, 2.0)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint16",
        "crs": crs_str,
        "transform": transform,
        "compress": "lzw",
    }

    with rasterio.open(ref_path, "w", **profile) as dst:
        dst.write(ref_arr, 1)

    with rasterio.open(tgt_path, "w", **profile) as dst:
        dst.write(tgt_arr, 1)

    print(f"[OK] Created uint16 Reference: {ref_path}")
    print(f"[OK] Created uint16 Target:    {tgt_path}")

    # Load default config and override with ORB detector backend
    config = load_config("config/phase1_config.yaml")
    config["keypoints"]["method"] = "ORB"
    config["keypoints"]["max_keypoints"] = 4000
    config["geospatial"]["crs_target"] = crs_str
    config.setdefault("evaluation", {})["min_inliers"] = 4
    config["evaluation"]["max_rmse"] = 15.0

    print("\nExecuting registration pipeline with ORB detector on uint16 raster...")
    summary = run_pipeline(
        reference_path=ref_path,
        target_path=tgt_path,
        output_path=output_path,
        config=config,
    )

    print("\n" + "=" * 60)
    print("  TEST RUN RESULTS (uint16 Single-Band ORB Input)")
    print("=" * 60)
    print(f"  Status           : {summary['status']}")
    print(f"  Runtime          : {summary['runtime_seconds']}s")
    print(f"  Detector Backend : {summary['features']['detector']}")
    print(f"  GCPs Collected   : {summary['features']['total_gcps_collected']}")
    print(f"  GCP Inliers      : {summary['features']['gcp_inliers']}")
    print(f"  Global RMSE (px) : {summary['metrics']['global_rmse_px']} px")
    print(f"  Spatial Coverage : {summary['metrics']['spatial_coverage'] * 100:.1f}%")
    print(f"  Output GeoTIFF   : {summary['files']['output']}")
    print("=" * 60)


if __name__ == "__main__":
    run_uint16_orb_test()
