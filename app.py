"""
app.py
~~~~~~
Copyright (c) 2026 Santhosh — MIT License

Streamlit Interactive Dashboard for Chandrayaan-2 Geospatial Image
Registration Pipeline. Reads configuration from config/pipeline_config.yaml
and runs the full registration pipeline with interactive parameter controls.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StreamlitApp")

CONFIG_PATH = ROOT / "config" / "pipeline_config.yaml"


def load_config() -> dict:
    """Load pipeline configuration from YAML."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as exc:
        st.error(f"Failed to load config: {exc}")
        return {}


# ─── Page Setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Yaazhi GeoAlign OS",
    page_icon="🌕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric label { color: #06b6d4!important; }
    .stMetric div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🌕 Yaazhi GeoAlign OS")
st.caption("Chandrayaan-2 Multi-Modal Geospatial Image Registration Platform — ISRO SIH26166")
st.divider()

# ─── Sidebar Config Panel ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pipeline Parameters")

    cfg = load_config()
    kp_cfg     = cfg.get("keypoints", {})
    tiling_cfg = cfg.get("tiling", {})
    ransac_cfg = cfg.get("ransac", {})
    eval_cfg   = cfg.get("evaluation", {})

    detector     = st.selectbox("Feature Detector", ["SIFT", "ORB"],
                                index=0 if kp_cfg.get("method", "SIFT") == "SIFT" else 1)
    max_kp       = st.slider("Max Keypoints / Tile", 500, 8000, kp_cfg.get("max_keypoints", 3000), 500)
    tile_size    = st.select_slider("Tile Size (px)", [256, 512, 768, 1024], value=tiling_cfg.get("tile_size", 512))
    overlap_pct  = st.slider("Tile Overlap", 0.0, 0.5, tiling_cfg.get("overlap_pct", 0.2), 0.05)
    ransac_thr   = st.slider("RANSAC Threshold (px)", 1.0, 15.0, ransac_cfg.get("threshold", 5.0), 0.5)
    min_inliers  = st.number_input("Min Inlier Count", 4, 500, eval_cfg.get("min_inliers", 4))

    st.divider()
    st.caption("📄 Config loaded from:")
    st.code(str(CONFIG_PATH.relative_to(ROOT)), language="bash")

# ─── Main Content ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📂 Input Image Pair")

    ref_file = st.file_uploader("Reference GeoTIFF (e.g., OHRC)", type=["tif", "tiff"],
                                key="ref_upload")
    tgt_file = st.file_uploader("Target GeoTIFF (e.g., TMC-2)", type=["tif", "tiff"],
                                key="tgt_upload")

    # Preset dataset loader
    st.subheader("📦 Or Use Preset Dataset")
    preset = st.selectbox("Select Preset", [
        "— Select —",
        "Synthetic Pair (demo)",
        "Hard 3-Band Scene (2048×2048)",
        "Real Sentinel-2 Satellite",
    ])

    preset_map = {
        "Synthetic Pair (demo)": (
            ROOT / "data" / "reference.tif",
            ROOT / "data" / "target.tif",
        ),
        "Hard 3-Band Scene (2048×2048)": (
            ROOT / "data" / "hard_scene" / "reference_2k.tif",
            ROOT / "data" / "hard_scene" / "target_2k.tif",
        ),
        "Real Sentinel-2 Satellite": (
            ROOT / "data" / "real_satellite" / "sentinel2_red_real.tif",
            ROOT / "data" / "real_satellite" / "sentinel2_transformed_real.tif",
        ),
    }

    if st.button("▶ Run Registration Pipeline", type="primary", use_container_width=True):
        ref_path, tgt_path = (None, None)

        if preset in preset_map:
            ref_path, tgt_path = preset_map[preset]
        elif ref_file and tgt_file:
            tmp_dir = ROOT / "data" / "tmp_upload"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            ref_path = tmp_dir / "uploaded_ref.tif"
            tgt_path = tmp_dir / "uploaded_tgt.tif"
            ref_path.write_bytes(ref_file.read())
            tgt_path.write_bytes(tgt_file.read())

        if not ref_path or not tgt_path:
            st.error("Please select a preset or upload both reference and target files.")
        elif not ref_path.exists() or not tgt_path.exists():
            st.error(f"Selected files do not yet exist locally. Run the data generators first.")
        else:
            from main import run_pipeline, load_config as _lc

            out_path = ROOT / "data" / "processed" / "streamlit_registered.tif"
            override_cfg = _lc(CONFIG_PATH)
            override_cfg["keypoints"]["method"]       = detector
            override_cfg["keypoints"]["max_keypoints"]= max_kp
            override_cfg["tiling"]["tile_size"]       = tile_size
            override_cfg["tiling"]["overlap_pct"]     = overlap_pct
            override_cfg["ransac"]["threshold"]       = ransac_thr
            override_cfg.setdefault("evaluation", {})["min_inliers"] = int(min_inliers)

            with st.spinner("Running registration pipeline..."):
                try:
                    summary = run_pipeline(ref_path, tgt_path, out_path, override_cfg)
                    st.session_state["summary"] = summary
                    st.success("Registration complete!")
                except Exception as exc:
                    logger.exception("Pipeline failed")
                    st.error(f"Pipeline error: {exc}")

with col_right:
    st.subheader("📊 Quality Gate Results")

    summary = st.session_state.get("summary")
    if summary:
        status_ok = summary.get("status") == "SUCCESS"
        st.metric("Overall Status",
                  "PASS" if status_ok else "FAIL",
                  delta="Quality Gate" if status_ok else "FAILED")

        m = summary.get("metrics", {})
        f = summary.get("features", {})
        c1, c2 = st.columns(2)
        c1.metric("Sub-Pixel RMSE",  f"{m.get('global_rmse_px', '--')} px")
        c2.metric("Spatial Coverage", f"{m.get('spatial_coverage', 0)*100:.1f}%")
        c3, c4 = st.columns(2)
        c3.metric("GCP Inliers", f.get("gcp_inliers", "--"))
        c4.metric("Runtime", f"{summary.get('runtime_seconds', '--')}s")

        st.subheader("🔢 Homography Matrix")
        H = summary.get("homography_matrix")
        if H:
            st.code(json.dumps(H, indent=2), language="json")

        st.subheader("📄 Full Summary JSON")
        with st.expander("View full JSON report"):
            st.json(summary)
    else:
        st.info("Run the registration pipeline to see results here.")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Copyright (c) 2026 Santhosh — MIT License | ISRO SIH26166 | Yaazhi GeoAlign OS")
