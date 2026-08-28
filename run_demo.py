"""
run_demo.py
~~~~~~~~~~~
1-Click Zero-Configuration Command Line Entrypoint.
Generates synthetic GeoTIFFs if needed and executes full registration pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.generate_synthetic_geotiff import create_synthetic_geotiffs
from main import run_pipeline, load_default_config


def main() -> None:
    print("\n" + "=" * 65)
    print("  [OK] YAAZHI GEOALIGN OS -- QUICK LAUNCHER")
    print("=" * 65)

    ref_path = Path("data/reference.tif")
    tgt_path = Path("data/target.tif")
    out_path = Path("outputs/registered_output.tif")

    if not ref_path.exists() or not tgt_path.exists():
        print("[1/2] Data files missing. Generating synthetic GeoTIFF pair...")
        create_synthetic_geotiffs()
    else:
        print("[1/2] Using existing GeoTIFF datasets in data/")

    print("[2/2] Running registration pipeline...")
    config = load_default_config()

    summary = run_pipeline(ref_path, tgt_path, out_path, config)

    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Overall Status   : {summary['status']}")
    print(f"  Runtime          : {summary['runtime_seconds']}s")
    print(f"  Sub-Pixel RMSE   : {summary['metrics']['global_rmse_px']} px")
    print(f"  Spatial Coverage : {summary['metrics']['spatial_coverage'] * 100:.1f}%")
    print(f"  GCP Inliers      : {summary['features']['gcp_inliers']}")
    print(f"  Output GeoTIFF   : {summary['files']['output']}")
    print("=" * 65)
    print("\n[TIP] Launch web interface anytime by double-clicking 'run.bat' or running 'py server.py'\n")


if __name__ == "__main__":
    main()
