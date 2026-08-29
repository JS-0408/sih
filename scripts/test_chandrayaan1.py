import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import run_pipeline, load_config

# Filter out the NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

def prepare_ch1_data():
    xml_path = Path("ch1_tmc_ncf_20090731T1812342475_d_img_d18/data/calibrated/20090731/ch1_tmc_ncf_20090731T1812342475_d_img_d18.xml")
    ref_out = Path("data/processed/ch1_reference.tif")
    tgt_out = Path("data/processed/ch1_target.tif")
    
    ref_out.parent.mkdir(parents=True, exist_ok=True)
    
    # We will extract a 2048 x 2048 window from the middle of the 158,034 pixel-tall strip
    crop_size = 2048
    
    print(f"[INFO] Reading massive PDS4 image via XML label: {xml_path}")
    with rasterio.open(xml_path) as src:
        # Calculate a window in the middle of the strip
        start_row = src.height // 2 - crop_size // 2
        start_col = src.width // 2 - crop_size // 2
        window = Window(start_col, start_row, crop_size, crop_size)
        
        print(f"[INFO] Extracting {crop_size}x{crop_size} window at row {start_row}, col {start_col}...")
        data = src.read(1, window=window)
        
        # This is raw data, we need to assign a dummy geospatial transform for the pipeline
        dummy_transform = from_origin(0.0, 0.0, 1.0, 1.0)
        dummy_crs = "EPSG:4326"
        
        profile = {
            "driver": "GTiff",
            "height": crop_size,
            "width": crop_size,
            "count": 1,
            "dtype": data.dtype,
            "crs": dummy_crs,
            "transform": dummy_transform,
            "compress": "lzw",
        }
        
        with rasterio.open(ref_out, "w", **profile) as dst:
            dst.write(data, 1)
            
    print(f"[OK] Saved reference crop to {ref_out}")
    
    # Now generate a geometrically distorted target
    print("[INFO] Generating geometrically shifted target (Rotation: 2.5 deg, Shift X: 15px, Shift Y: -20px, Scale: 1.01)")
    h, w = data.shape
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle=2.5, scale=1.01)
    rot_mat[0, 2] += 15.0
    rot_mat[1, 2] += -20.0
    
    # We convert to float32 for warpAffine to avoid clipping weirdness on uint16, then cast back
    data_f = data.astype(np.float32)
    tgt_arr = cv2.warpAffine(data_f, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
    tgt_arr = tgt_arr.astype(data.dtype)
    
    with rasterio.open(tgt_out, "w", **profile) as dst:
        dst.write(tgt_arr, 1)
        
    print(f"[OK] Saved distorted target to {tgt_out}")
    return ref_out, tgt_out, dummy_crs

def main():
    ref_path, tgt_path, crs_str = prepare_ch1_data()
    out_path = Path("outputs/ch1_registered.tif")
    
    config = load_config("config/phase1_config.yaml")
    # Tweak config for Chandrayaan-1's uint16 high-dynamic range
    config["tiling"]["tile_size"] = 512
    config.setdefault("preprocessing", {})["mode"] = "log_clahe"  # Best for raw lunar imagery
    config.setdefault("ransac", {})["model"] = "affine"
    config["keypoints"]["method"] = "SIFT"
    config["keypoints"]["max_keypoints"] = 5000
    config["geospatial"]["crs_target"] = crs_str
    config.setdefault("evaluation", {})["min_inliers"] = 15
    config["evaluation"]["max_rmse"] = 25.0
    
    print("\n" + "=" * 65)
    print("  [OK] EXECUTING PIPELINE ON CHANDRAYAAN-1 TMC IMAGERY")
    print("=" * 65)
    
    summary = run_pipeline(ref_path, tgt_path, out_path, config)
    
    print("\n" + "=" * 65)
    print("  CHANDRAYAAN-1 TEST RESULTS")
    print("=" * 65)
    print(f"  Overall Status   : {summary['status']}")
    print(f"  Runtime          : {summary['runtime_seconds']}s")
    print(f"  GCP Inliers      : {summary['features']['gcp_inliers']}")
    print(f"  Global RMSE (px) : {summary['metrics']['fitting_rmse_px']} px")
    print(f"  Spatial Coverage : {summary['metrics']['spatial_coverage'] * 100:.1f}%")
    print(f"  Output GeoTIFF   : {summary['files']['output']}")
    print("=" * 65)

if __name__ == "__main__":
    main()
