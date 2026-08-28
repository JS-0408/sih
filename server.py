"""
server.py
~~~~~~~~~
Production Web Application Server & API Backend for Geospatial Registration Dashboard.

Serves the interactive web user interface and exposes REST API endpoints for:
- Executing registration jobs on selected or uploaded GeoTIFF pairs
- Generating PNG preview overlays for visual inspection
- Displaying real-time Quality Gate metrics and downloadable output files
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, send_file

from main import run_pipeline, load_config
from src.io.raster_loader import RasterLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GeoServer")

app = Flask(__name__, static_folder="web", static_url_path="")
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


@app.route("/")
def index():
    """Serve the web UI frontend."""
    return send_from_directory("web", "index.html")


@app.route("/api/datasets", methods=["GET"])
def get_datasets():
    """List available GeoTIFF reference/target test pairs in data/ directory."""
    datasets = []
    
    # Standard synthetic pair
    if (DATA_DIR / "reference.tif").exists() and (DATA_DIR / "target.tif").exists():
        datasets.append({
            "id": "synthetic",
            "name": "Standard Synthetic Pair (1024x1024 uint8)",
            "reference": str(DATA_DIR / "reference.tif"),
            "target": str(DATA_DIR / "target.tif"),
        })

    # Hard 2K pair
    hard_ref = DATA_DIR / "hard_scene" / "reference_2k.tif"
    hard_tgt = DATA_DIR / "hard_scene" / "target_2k.tif"
    if hard_ref.exists() and hard_tgt.exists():
        datasets.append({
            "id": "hard_2k",
            "name": "Hard 3-Band Scene (2048x2048 RGB, EPSG:32643)",
            "reference": str(hard_ref),
            "target": str(hard_tgt),
        })

    # Real Sentinel-2 satellite pair
    real_ref = DATA_DIR / "real_satellite" / "sentinel2_red_real.tif"
    real_tgt = DATA_DIR / "real_satellite" / "sentinel2_transformed_real.tif"
    if real_ref.exists() and real_tgt.exists():
        datasets.append({
            "id": "real_sentinel2",
            "name": "Real Sentinel-2 Satellite Raster (1536x1536, EPSG:32610)",
            "reference": str(real_ref),
            "target": str(real_tgt),
        })

    return jsonify({"datasets": datasets})


@app.route("/api/register", methods=["POST"])
def register_pipeline():
    """Execute registration pipeline with parameter payload from Web UI."""
    payload = request.get_json(force=True) or {}
    
    ref_path = payload.get("reference")
    tgt_path = payload.get("target")
    detector = payload.get("detector", "SIFT")
    tile_size = int(payload.get("tile_size", 512))
    max_keypoints = int(payload.get("max_keypoints", 3000))
    ransac_threshold = float(payload.get("ransac_threshold", 5.0))

    if not ref_path or not tgt_path:
        return jsonify({"error": "Missing reference or target file path"}), 400

    out_name = f"web_registered_{detector.lower()}.tif"
    out_path = OUTPUTS_DIR / out_name

    import rasterio
    with rasterio.open(ref_path) as ds:
        ref_crs = str(ds.crs) if ds.crs else "EPSG:4326"

    cfg = load_config(BASE_DIR / "config" / "phase1_config.yaml")
    cfg["keypoints"]["method"] = detector
    cfg["keypoints"]["max_keypoints"] = max_keypoints
    cfg["tiling"]["tile_size"] = tile_size
    cfg["ransac"]["threshold"] = ransac_threshold
    cfg["geospatial"]["crs_target"] = ref_crs
    cfg.setdefault("evaluation", {})["min_inliers"] = 4
    cfg["evaluation"]["max_rmse"] = 15.0

    try:
        summary = run_pipeline(ref_path, tgt_path, out_path, cfg)
        summary["output_preview_url"] = f"/api/preview?path={out_path}"
        summary["reference_preview_url"] = f"/api/preview?path={ref_path}"
        summary["target_preview_url"] = f"/api/preview?path={tgt_path}"
        return jsonify(summary)
    except Exception as exc:
        logger.error(f"Registration API error: {exc}", exc_info=True)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/preview", methods=["GET"])
def get_image_preview():
    """Render 8-bit PNG preview of any GeoTIFF raster for Web UI displaying."""
    path_str = request.args.get("path")
    if not path_str or not Path(path_str).exists():
        return jsonify({"error": "File not found"}), 404

    try:
        loader = RasterLoader()
        arr, _ = loader.load(path_str)

        # Convert to (H, W, C) uint8 for browser preview
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3, 4):
                img = np.moveaxis(arr, 0, -1)
            else:
                img = arr
        else:
            img = arr

        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]

        # Normalise to 8-bit uint8
        if img.dtype != np.uint8:
            lo, hi = img.min(), img.max()
            if hi > lo:
                img = ((img - lo) / (hi - lo) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)

        # Resize for fast network preview transfer if > 1024
        h, w = img.shape[:2]
        if max(h, w) > 1024:
            scale = 1024.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        is_success, buffer = cv2.imencode(".png", img)
        if not is_success:
            return jsonify({"error": "Encoding error"}), 500

        return send_file(io.BytesIO(buffer), mimetype="image/png")
    except Exception as exc:
        logger.error(f"Preview generation error: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/outputs/<path:filename>")
def serve_output_file(filename: str):
    """Serve output files for download."""
    return send_from_directory(OUTPUTS_DIR, filename)


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  [OK] GEOSPATIAL REGISTRATION PLATFORM - WEB DASHBOARD READY")
    print("  [URL] Access UI at: http://127.0.0.1:5000")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
