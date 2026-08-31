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
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
from flask import Flask, jsonify, request, send_from_directory, send_file

from main import run_pipeline, load_config
from src.io.raster_loader import RasterLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GeoServer")

app = Flask(__name__, static_folder="web", static_url_path="")
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
TMP_DIR     = DATA_DIR / "tmp_upload"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline status codes that count as PASS
_PASS_STATUSES = {"SUCCESS", "LOW_CONFIDENCE"}


def _ensure_raster_crs(file_path: Path, work_dir: Path, default_crs: str = "EPSG:4326") -> tuple[Path, str]:
    """
    Ensure the raster can be opened by rasterio and has a valid CRS.
    Supports PDS4 .img files by finding their matching .xml label in the workspace.
    Handles large un-georeferenced or PDS4 rasters by generating a GeoTIFF with CRS.
    """
    path = file_path

    # Try opening directly; if it fails (e.g. .img without label), resolve .xml match in workspace
    try:
        with rasterio.open(str(path)) as ds:
            pass
    except Exception:
        stem = path.stem
        xml_matches = list(BASE_DIR.rglob(f"{stem}.xml"))
        if xml_matches:
            path = xml_matches[0]
            logger.info(f"Resolved raw file {file_path.name} -> PDS label {path}")

    # Now open path and ensure valid GeoTIFF format with CRS
    with rasterio.open(str(path)) as src:
        crs  = src.crs
        w, h = src.width, src.height

        need_convert = (crs is None) or (path.suffix.lower() == ".xml") or (max(w, h) > 4096)

        if not need_convert:
            return path, str(crs)

        # Convert / crop to RAM-safe GeoTIFF with valid CRS
        out_path = work_dir / f"georef_{path.stem}.tif"
        crop_w   = min(2048, w)
        crop_h   = min(2048, h)
        row_off  = 10000 if h > 12000 else 0
        window   = rasterio.windows.Window(0, row_off, crop_w, crop_h)
        data     = src.read(window=window)

        if data.ndim == 2:
            data = data[np.newaxis, ...]

        profile = {
            "driver":    "GTiff",
            "height":    crop_h,
            "width":     crop_w,
            "count":     data.shape[0],
            "dtype":     str(data.dtype),
            "crs":       default_crs,
            "transform": rasterio.transform.from_origin(0, 0, 1, 1),
        }
        with rasterio.open(str(out_path), "w", **profile) as dst:
            dst.write(data)

        return out_path, default_crs



# ─── Static / UI routes ───────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the web UI frontend."""
    return send_from_directory("web", "index.html")


# ─── Dataset listing ──────────────────────────────────────────────────────────

@app.route("/api/datasets", methods=["GET"])
def get_datasets():
    """List available GeoTIFF reference/target test pairs in data/ directory."""
    datasets = []

    # 1. Standard synthetic pair
    if (DATA_DIR / "reference.tif").exists() and (DATA_DIR / "target.tif").exists():
        datasets.append({
            "id": "synthetic",
            "name": "Standard Synthetic Pair (1024×1024 uint8)",
            "reference": str(DATA_DIR / "reference.tif"),
            "target":    str(DATA_DIR / "target.tif"),
        })

    # 2. Hard 2K pair
    hard_ref = DATA_DIR / "hard_scene" / "reference_2k.tif"
    hard_tgt = DATA_DIR / "hard_scene" / "target_2k.tif"
    if hard_ref.exists() and hard_tgt.exists():
        datasets.append({
            "id": "hard_2k",
            "name": "Hard 3-Band Scene (2048×2048 RGB, EPSG:32643)",
            "reference": str(hard_ref),
            "target":    str(hard_tgt),
        })

    # 3. Real Sentinel-2 satellite pair
    real_ref = DATA_DIR / "real_satellite" / "sentinel2_red_real.tif"
    real_tgt = DATA_DIR / "real_satellite" / "sentinel2_transformed_real.tif"
    if real_ref.exists() and real_tgt.exists():
        datasets.append({
            "id": "real_sentinel2",
            "name": "Real Sentinel-2 Satellite Raster (1536×1536, EPSG:32610)",
            "reference": str(real_ref),
            "target":    str(real_tgt),
        })

    return jsonify({"datasets": datasets})


# ─── Registration Preset endpoint ──────────────────────────────────────────────

@app.route("/api/register-preset", methods=["GET", "POST"])
def register_preset():
    """
    Run registration pipeline on a server-side dataset preset (no upload needed).
    Accepts JSON: { "reference": "...", "target": "...", "detector": "SIFT", "model": "homography", "use_deep": false }
    """
    payload  = request.get_json(force=True) or {}
    ref_path = Path(payload.get("reference", ""))
    tgt_path = Path(payload.get("target", ""))

    if not ref_path.exists():
        return jsonify({"error": f"Reference file not found: {ref_path}"}), 400
    if not tgt_path.exists():
        return jsonify({"error": f"Target file not found: {tgt_path}"}), 400

    detector = payload.get("detector", "SIFT").upper()
    model    = payload.get("model", "homography").lower()
    use_deep = bool(payload.get("use_deep", False))
    run_id   = uuid.uuid4().hex[:8]
    run_tmp  = TMP_DIR / run_id
    run_tmp.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / f"preset_registered_{run_id}.tif"

    try:
        proc_ref, ref_crs = _ensure_raster_crs(ref_path, run_tmp)
        proc_tgt, _       = _ensure_raster_crs(tgt_path, run_tmp, default_crs=ref_crs)

        with rasterio.open(str(proc_ref)) as ds:
            ref_w, ref_h = ds.width, ds.height

        cfg_path = BASE_DIR / "config" / "pipeline_config.yaml"
        if not cfg_path.exists():
            cfg_path = BASE_DIR / "config" / "phase1_config.yaml"
        cfg = load_config(cfg_path)

        cfg.setdefault("keypoints",     {})["method"]      = detector
        cfg["keypoints"].setdefault("max_keypoints", 3000)
        cfg.setdefault("tiling",        {}).setdefault("tile_size", 512)
        cfg.setdefault("ransac",        {})["model"]        = model
        cfg.setdefault("geospatial",    {})["crs_target"]   = ref_crs
        cfg.setdefault("evaluation",    {})["min_inliers"]  = 4
        cfg["evaluation"]["max_rmse"]                       = 15.0
        cfg.setdefault("deep_matching", {})["enabled"]      = use_deep

        t0      = time.time()
        summary = run_pipeline(str(proc_ref), str(proc_tgt), str(out_path), cfg)
        elapsed = round(time.time() - t0, 3)

        return jsonify(_build_frontend_response(summary, elapsed, detector, model, ref_w, ref_h, proc_ref, out_path, ref_path.name, tgt_path.name, cfg))

    except Exception as exc:
        logger.error(f"Preset registration error: {exc}", exc_info=True)
        return jsonify({"error": str(exc)}), 500
    finally:
        shutil.rmtree(run_tmp, ignore_errors=True)


# ─── Upload Registration endpoint ─────────────────────────────────────────────

@app.route("/api/register", methods=["GET", "POST"])
def register_pipeline():
    """
    Execute registration pipeline with uploaded files.
    Accepts multipart/form-data with files[] (2 files).
    """
    uploaded_files = request.files.getlist("files[]")
    if len(uploaded_files) < 2:
        return jsonify({"error": "At least two raster files are required (reference + target)."}), 400

    run_id  = uuid.uuid4().hex[:8]
    run_tmp = TMP_DIR / run_id
    run_tmp.mkdir(parents=True, exist_ok=True)

    try:
        ref_filename = uploaded_files[0].filename or "reference.tif"
        tgt_filename = uploaded_files[1].filename or "target.tif"
        raw_ref      = run_tmp / ref_filename
        raw_tgt      = run_tmp / tgt_filename

        uploaded_files[0].save(str(raw_ref))
        uploaded_files[1].save(str(raw_tgt))

        detector = request.form.get("detector", "SIFT").upper()
        model    = request.form.get("model", "homography").lower()
        use_deep = request.form.get("use_deep", "false").lower() in ("true", "1", "yes")
        out_path = OUTPUTS_DIR / f"web_registered_{run_id}.tif"

        proc_ref, ref_crs = _ensure_raster_crs(raw_ref, run_tmp)
        proc_tgt, _       = _ensure_raster_crs(raw_tgt, run_tmp, default_crs=ref_crs)

        with rasterio.open(str(proc_ref)) as ds:
            ref_w, ref_h = ds.width, ds.height

        cfg_path = BASE_DIR / "config" / "pipeline_config.yaml"
        if not cfg_path.exists():
            cfg_path = BASE_DIR / "config" / "phase1_config.yaml"
        cfg = load_config(cfg_path)

        cfg.setdefault("keypoints",     {})["method"]      = detector
        cfg["keypoints"].setdefault("max_keypoints", 3000)
        cfg.setdefault("tiling",        {}).setdefault("tile_size", 512)
        cfg.setdefault("ransac",        {})["model"]        = model
        cfg.setdefault("geospatial",    {})["crs_target"]   = ref_crs
        cfg.setdefault("evaluation",    {})["min_inliers"]  = 4
        cfg["evaluation"]["max_rmse"]                       = 15.0
        cfg.setdefault("deep_matching", {})["enabled"]      = use_deep

        t0      = time.time()
        summary = run_pipeline(str(proc_ref), str(proc_tgt), str(out_path), cfg)
        elapsed = round(time.time() - t0, 3)

        # Preserve reference raster in OUTPUTS_DIR for browser preview
        saved_ref = OUTPUTS_DIR / f"web_ref_{run_id}.tif"
        shutil.copy2(proc_ref, saved_ref)

        return jsonify(_build_frontend_response(summary, elapsed, detector, model, ref_w, ref_h, saved_ref, out_path, ref_filename, tgt_filename, cfg))

    except Exception as exc:
        logger.error(f"Registration API error: {exc}", exc_info=True)
        return jsonify({"error": str(exc)}), 500

    finally:
        shutil.rmtree(run_tmp, ignore_errors=True)


# ─── Response Builder Helper ──────────────────────────────────────────────────

def _build_frontend_response(
    summary: dict, elapsed: float, detector: str, model: str,
    ref_w: int, ref_h: int, ref_path: Path, out_path: Path,
    ref_name: str, tgt_name: str, cfg: dict
) -> dict:
    pipe_status = summary.get("status", "UNKNOWN")
    is_pass     = pipe_status in _PASS_STATUSES
    metrics     = summary.get("metrics", {})
    features    = summary.get("features", {})
    tiling      = summary.get("tiling", {})

    rows_n, cols_n = 1, 1
    grid_str = str(tiling.get("grid", ""))
    if "x" in grid_str:
        try:
            parts = grid_str.split("x")
            rows_n, cols_n = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            pass

    eval_report    = summary.get("evaluation", {})
    tile_summaries = eval_report.get("tile_summaries", [])
    tile_density   = (
        [float(t.get("inlier_ratio", 0.0)) for t in tile_summaries]
        if tile_summaries else [0.0] * (rows_n * cols_n)
    )

    out_resolution = f"{ref_w}x{ref_h}"
    if out_path.exists():
        try:
            with rasterio.open(str(out_path)) as ds:
                out_resolution = f"{ds.width}x{ds.height}"
        except Exception:
            pass

    preview_ref_url = f"/api/preview?path={ref_path}"
    preview_reg_url = f"/api/preview?path={out_path}" if out_path.exists() else None
    sensor_pair     = _sensor_pair_from_names(ref_name, tgt_name)
    log_lines       = _build_log(summary, pipe_status, elapsed, rows_n, cols_n, features, metrics)

    return {
        "status":              "PASS" if is_pass else "FAIL",
        "rmse":                round(metrics.get("fitting_rmse_px") or 0.0, 4),
        "coverage_pct":        round((metrics.get("spatial_coverage") or 0.0) * 100, 1),
        "inliers":             features.get("gcp_inliers", 0),
        "elapsed_s":           summary.get("runtime_seconds", elapsed),
        "detector":            features.get("detector", detector),
        "model":               features.get("gcp_model", model),
        "tile_grid":           {"rows": rows_n, "cols": cols_n},
        "tile_inlier_density": tile_density,
        "keypoints_ref":       features.get("total_gcps_collected"),
        "keypoints_tgt":       features.get("total_gcps_collected"),
        "match_ratio":         cfg.get("flann", {}).get("ratio_threshold", 0.75),
        "min_inliers":         cfg["evaluation"]["min_inliers"],
        "max_rmse":            cfg["evaluation"]["max_rmse"],
        "output_resolution":   out_resolution,
        "sensor_pair":         sensor_pair,
        "preview_ref_url":     preview_ref_url,
        "preview_reg_url":     preview_reg_url,
        "log":                 log_lines,
        "output_preview_url":      preview_reg_url,
        "reference_preview_url":   preview_ref_url,
        "files":                   summary.get("files", {}),
        "homography_matrix":       summary.get("homography_matrix"),
        "warnings":                summary.get("warnings", []),
        "independent_validation":  summary.get("independent_validation", {}),
    }


def _sensor_pair_from_names(ref_name: str, tgt_name: str) -> str:
    def _detect(name: str) -> str:
        lower = name.lower()
        if "_ohr_" in lower or lower.startswith("ch2_ohr"):
            return "OHRC"
        if "_tmc_" in lower or lower.startswith("ch2_tmc") or lower.startswith("ch1_tmc"):
            return "TMC-2"
        if "_iir_" in lower or lower.startswith("ch2_iir"):
            return "IIRS"
        if lower.startswith("s2a_") or lower.startswith("s2b_"):
            return "Sentinel-2"
        return Path(name).stem[:8] or "Sensor"
    return f"{_detect(ref_name)} → {_detect(tgt_name)}"


def _build_log(summary: dict, status: str, elapsed: float, rows_n: int, cols_n: int, features: dict, metrics: dict) -> list[str]:
    tiling = summary.get("tiling", {})
    lines = [
        f"Reference raster loaded: {summary.get('files', {}).get('reference', '—')}",
        f"Target raster loaded:    {summary.get('files', {}).get('target', '—')}",
        f"Preprocessing mode: {summary.get('preprocessing', {}).get('mode', '—')}",
        f"Tile grid: {rows_n}×{cols_n} tiles (size={tiling.get('tile_size', '—')}px, overlap={tiling.get('overlap', '—')})",
        f"Feature detector: {features.get('detector', '—')}",
        f"GCPs collected across all tiles: {features.get('total_gcps_collected', '—')}",
        f"Sub-pixel refinement: {features.get('n_subpixel_refined', '—')} correspondences refined",
        f"GCP inliers (RANSAC): {features.get('gcp_inliers', '—')}",
        f"Fitting RMSE: {metrics.get('fitting_rmse_px', '—')} px",
        f"Spatial coverage: {round((metrics.get('spatial_coverage') or 0.0) * 100, 1)}%",
    ]
    val = summary.get("independent_validation", {})
    if val:
        valid_str = "VALID" if val.get("subpixel_claim_valid") else "FLAGGED"
        lines.append(f"Independent validation P95: {val.get('p95_px', '—')} px ({valid_str})")
    for w in summary.get("warnings", []):
        lines.append(f"⚠  {w}")
    lines.append(f"Pipeline completed in {elapsed}s — Status: {status}")
    return lines


# ─── Preview endpoint ─────────────────────────────────────────────────────────

@app.route("/api/preview", methods=["GET"])
def get_image_preview():
    """Render an 8-bit PNG preview of any GeoTIFF path for the browser."""
    path_str = request.args.get("path")
    if not path_str or not Path(path_str).exists():
        return jsonify({"error": "File not found"}), 404

    try:
        with rasterio.open(path_str) as ds:
            arr = ds.read()

        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            img = np.moveaxis(arr, 0, -1)
        else:
            img = arr

        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]

        if img.dtype != np.uint8:
            lo, hi = img.min(), img.max()
            img = ((img - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros_like(img, dtype=np.uint8)

        h, w = img.shape[:2]
        if max(h, w) > 1024:
            scale = 1024.0 / max(h, w)
            img   = cv2.resize(img, (int(w * scale), int(h * scale)))

        ok, buf = cv2.imencode(".png", img)
        if not ok:
            return jsonify({"error": "PNG encoding failed"}), 500

        return send_file(io.BytesIO(buf), mimetype="image/png")

    except Exception as exc:
        logger.error(f"Preview generation error: {exc}")
        return jsonify({"error": str(exc)}), 500


# ─── Output file serving ──────────────────────────────────────────────────────

@app.route("/outputs/<path:filename>")
def serve_output_file(filename: str):
    """Serve output files (registered GeoTIFFs) for download."""
    return send_from_directory(OUTPUTS_DIR, filename)


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  [OK] GEOSPATIAL REGISTRATION PLATFORM - WEB DASHBOARD READY")
    print("  [URL] Access UI at: http://127.0.0.1:5000")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
