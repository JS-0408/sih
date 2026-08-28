"""
app.py
~~~~~~~
Copyright (c) 2026 Santhosh Jayakumar & Team — MIT License
Part of Yaazhi GeoAlign OS / ISRO SIH26166 Chandrayaan-2 Registration System.

Streamlit Interactive Dashboard:
 - 2D Split-Curtain swipe comparison (before/after registration).
 - Interactive 3D Terrain Viewer (PyVista HTML mesh in embedded iframe).
 - Pipeline parameters read from config/pipeline_config.yaml.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import base64
import io
import json
import logging
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StreamlitApp")

CONFIG_PATH = ROOT / "config" / "pipeline_config.yaml"

# ─── Page Setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Yaazhi GeoAlign OS — Chandrayaan-2",
    page_icon="🌕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #0b0f19; }
section[data-testid="stSidebar"] { background: rgba(18,24,38,0.95); }
.stMetric label { color: #06b6d4 !important; font-size: 0.78rem; }
.stMetric div[data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 700; color: #fff; }
.section-title {
    color: #06b6d4; font-size: 1.05rem; font-weight: 600;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 0.4rem; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Config Loader ────────────────────────────────────────────────────────────

def load_config() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as exc:
        st.error(f"Config load failed: {exc}")
        return {}


# ─── Helper: image → base64 PNG for embedding ────────────────────────────────

def _arr_to_b64_png(arr: np.ndarray) -> str:
    import cv2
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.dtype != np.uint8:
        lo, hi = arr.min(), arr.max()
        arr = ((arr - lo) / max(hi - lo, 1) * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        return ""
    return base64.b64encode(buf).decode("utf-8")


# ─── Split curtain comparison widget ─────────────────────────────────────────

def split_curtain_widget(before_b64: str, after_b64: str) -> None:
    """Render an interactive CSS split-curtain comparison in Streamlit."""
    html = f"""
<style>
.curtain-wrap {{
  position:relative; width:100%; height:520px;
  overflow:hidden; border-radius:12px;
  border:1px solid rgba(255,255,255,0.1);
  background:#000;
  user-select:none;
}}
.curtain-wrap img {{
  position:absolute; top:0; left:0;
  width:100%; height:100%; object-fit:contain;
}}
.img-after-wrap {{
  position:absolute; top:0; left:0;
  width:50%; height:100%; overflow:hidden;
  border-right:2px solid #06b6d4;
  z-index:2;
}}
.img-after-wrap img {{ width:100%; max-width:none; height:100%; object-fit:contain; }}
.handle {{
  position:absolute; top:0; bottom:0;
  left:50%; width:4px; z-index:10;
  cursor:ew-resize;
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  transform:translateX(-50%);
}}
.handle-line {{ flex:1; width:2px; background:#06b6d4; box-shadow:0 0 8px #06b6d4; }}
.handle-circle {{
  width:36px; height:36px; border-radius:50%;
  background:#06b6d4; color:#000;
  display:flex; align-items:center; justify-content:center;
  font-size:1.1rem; font-weight:700;
  box-shadow:0 0 12px rgba(6,182,212,0.8);
}}
.badge {{
  position:absolute; bottom:10px;
  padding:3px 10px; border-radius:6px;
  background:rgba(0,0,0,0.6); color:#fff; font-size:0.72rem;
  border:1px solid rgba(255,255,255,0.15); z-index:15;
}}
.badge-left {{ left:10px; }}
.badge-right {{ right:10px; }}
</style>
<div class="curtain-wrap" id="cwrap">
  <img src="data:image/png;base64,{before_b64}" id="img-before" alt="Before">
  <div class="img-after-wrap" id="after-wrap">
    <img src="data:image/png;base64,{after_b64}" id="img-after" alt="After">
  </div>
  <div class="handle" id="handle">
    <div class="handle-line"></div>
    <div class="handle-circle">⇔</div>
    <div class="handle-line"></div>
  </div>
  <div class="badge badge-left">Target (Before)</div>
  <div class="badge badge-right">Registered (After)</div>
</div>
<script>
(function() {{
  const wrap   = document.getElementById('cwrap');
  const after  = document.getElementById('after-wrap');
  const handle = document.getElementById('handle');
  const imgAft = document.getElementById('img-after');
  let drag = false;
  function setPos(clientX) {{
    const r = wrap.getBoundingClientRect();
    let x = Math.min(Math.max(clientX - r.left, 0), r.width);
    const pct = (x / r.width * 100).toFixed(2);
    after.style.width = pct + '%';
    handle.style.left = pct + '%';
    imgAft.style.width = r.width + 'px';
  }}
  wrap.addEventListener('mousedown', e => {{ drag=true; setPos(e.clientX); }});
  window.addEventListener('mousemove', e => {{ if(drag) setPos(e.clientX); }});
  window.addEventListener('mouseup',   () => {{ drag=false; }});
  wrap.addEventListener('touchstart',  e => setPos(e.touches[0].clientX), {{passive:true}});
  window.addEventListener('touchmove', e => {{ if(drag) setPos(e.touches[0].clientX); }}, {{passive:true}});
  // Init at 50%
  const r = wrap.getBoundingClientRect();
  imgAft.style.width = r.width + 'px';
}})();
</script>
"""
    st.components.v1.html(html, height=540, scrolling=False)


# ─── 3D Terrain Viewer ────────────────────────────────────────────────────────

def terrain_3d_viewer(html_path: Path | None, height: int = 520) -> None:
    """Embed a PyVista-exported HTML terrain mesh in a Streamlit iframe."""
    if html_path is None or not html_path.exists():
        st.info("No 3D terrain file found. Run the pipeline first, or generate a synthetic DEM below.")
        if st.button("Generate Synthetic Lunar DEM (Demo)", key="gen_dem"):
            from src.geometry.mesh_builder import SurfaceMeshBuilder
            builder = SurfaceMeshBuilder(z_scale=float(st.session_state.get("z_scale", 2.0)),
                                         downsample=int(st.session_state.get("down_factor", 4)))
            dem = builder.generate_synthetic_dem(width=256, height=256)
            out = ROOT / "data" / "processed" / "terrain_3d.html"
            builder.build_and_export(elevation=dem, output_path=out)
            st.session_state["terrain_html"] = out
            st.rerun()
        return

    content = html_path.read_text(encoding="utf-8")
    b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    iframe_src = f"data:text/html;base64,{b64}"
    st.components.v1.iframe(iframe_src, height=height, scrolling=True)


# ─── Header ───────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.markdown("<div style='font-size:2.5rem;margin-top:5px;'>🌕</div>", unsafe_allow_html=True)
with col_title:
    st.title("Yaazhi GeoAlign OS")
    st.caption("Chandrayaan-2 Multi-Modal Geospatial Image Registration — ISRO SIH26166")
st.divider()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">⚙️ Pipeline Parameters</div>', unsafe_allow_html=True)

    cfg        = load_config()
    kp_cfg     = cfg.get("keypoints", {})
    tiling_cfg = cfg.get("tiling", {})
    ransac_cfg = cfg.get("ransac", {})
    eval_cfg   = cfg.get("evaluation", {})
    mesh_cfg   = cfg.get("mesh_builder", {})

    detector    = st.selectbox("Feature Detector", ["SIFT", "ORB"],
                               index=0 if kp_cfg.get("method","SIFT")=="SIFT" else 1)
    max_kp      = st.slider("Max Keypoints / Tile", 500, 8000, kp_cfg.get("max_keypoints", 3000), 500)
    tile_size   = st.select_slider("Tile Size (px)", [256, 512, 768, 1024],
                                   value=tiling_cfg.get("tile_size", 512))
    overlap_pct = st.slider("Tile Overlap", 0.0, 0.5, tiling_cfg.get("overlap_pct", 0.2), 0.05)
    ransac_thr  = st.slider("RANSAC Threshold (px)", 1.0, 15.0, ransac_cfg.get("threshold", 5.0), 0.5)

    st.divider()
    st.markdown('<div class="section-title">🏔️ 3D Terrain Parameters</div>', unsafe_allow_html=True)
    z_scale     = st.slider("Vertical Exaggeration (Z-scale)", 0.5, 10.0,
                             mesh_cfg.get("z_scale", 2.0), 0.5, key="z_scale")
    down_factor = st.select_slider("Spatial Downsampling", [1, 2, 4, 8],
                                    value=mesh_cfg.get("downsample", 4), key="down_factor")

    st.divider()
    st.markdown('<div class="section-title">📦 Preset Datasets</div>', unsafe_allow_html=True)
    preset = st.selectbox("Load Preset", [
        "— Select —",
        "Synthetic Pair (demo)",
        "Hard 3-Band Scene (2048×2048)",
        "Real Sentinel-2 Satellite",
    ])


# ─── Main Tabs ────────────────────────────────────────────────────────────────
tab_register, tab_3d = st.tabs(["🛰️ Registration & 2D Comparison", "🏔️ 3D Terrain Viewer"])


# ─── Tab 1: Registration ──────────────────────────────────────────────────────
with tab_register:
    col_ctrl, col_result = st.columns([2, 3])

    with col_ctrl:
        st.markdown('<div class="section-title">📂 Input Files</div>', unsafe_allow_html=True)

        ref_file = st.file_uploader("Reference GeoTIFF", type=["tif", "tiff"], key="ref")
        tgt_file = st.file_uploader("Target GeoTIFF",    type=["tif", "tiff"], key="tgt")

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

        run_clicked = st.button("▶ Run Registration Pipeline", type="primary",
                                use_container_width=True, key="run_btn")

        if run_clicked:
            ref_path, tgt_path = None, None

            if preset in preset_map:
                ref_path, tgt_path = preset_map[preset]
            elif ref_file and tgt_file:
                tmp = ROOT / "data" / "tmp_upload"
                tmp.mkdir(parents=True, exist_ok=True)
                ref_path = tmp / "uploaded_ref.tif"
                tgt_path = tmp / "uploaded_tgt.tif"
                ref_path.write_bytes(ref_file.read())
                tgt_path.write_bytes(tgt_file.read())

            if not ref_path or not tgt_path:
                st.error("Select a preset or upload both files.")
            elif not ref_path.exists() or not tgt_path.exists():
                st.error("Selected files not found locally. Run data generators first.")
            else:
                from main import run_pipeline, load_config as _lc
                out_path    = ROOT / "data" / "processed" / "streamlit_registered.tif"
                override_cfg = _lc(CONFIG_PATH)
                override_cfg["keypoints"]["method"]       = detector
                override_cfg["keypoints"]["max_keypoints"]= max_kp
                override_cfg["tiling"]["tile_size"]       = tile_size
                override_cfg["tiling"]["overlap_pct"]     = overlap_pct
                override_cfg["ransac"]["threshold"]       = ransac_thr
                override_cfg.setdefault("evaluation", {})["min_inliers"] = 4

                with st.spinner("Running pipeline..."):
                    try:
                        summary = run_pipeline(ref_path, tgt_path, out_path, override_cfg)
                        st.session_state["summary"]      = summary
                        st.session_state["ref_path"]     = str(ref_path)
                        st.session_state["out_path"]     = str(out_path)
                        st.success("Pipeline complete!")
                    except Exception as exc:
                        logger.exception("Pipeline failed")
                        st.error(f"Error: {exc}")

    with col_result:
        st.markdown('<div class="section-title">📊 Quality Gate Metrics</div>', unsafe_allow_html=True)

        summary = st.session_state.get("summary")
        if summary:
            status_ok = summary.get("status") == "SUCCESS"
            st.metric("Overall Status", "PASS ✅" if status_ok else "FAIL ❌")

            m  = summary.get("metrics", {})
            f  = summary.get("features", {})
            c1, c2 = st.columns(2)
            c1.metric("Sub-Pixel RMSE",    f"{m.get('global_rmse_px','--')} px")
            c2.metric("Spatial Coverage",  f"{m.get('spatial_coverage',0)*100:.1f}%")
            c3, c4 = st.columns(2)
            c3.metric("GCP Inliers",       f"{f.get('gcp_inliers','--')}")
            c4.metric("Runtime",           f"{summary.get('runtime_seconds','--')}s")

            H = summary.get("homography_matrix")
            if H:
                st.markdown('<div class="section-title">🔢 Homography Matrix</div>',
                            unsafe_allow_html=True)
                st.code(json.dumps(H, indent=2), language="json")

            with st.expander("📄 Full Summary JSON"):
                st.json(summary)
        else:
            st.info("Run the pipeline to view metrics.")

    # ── 2D Split Curtain ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">👁️ Visual Split Alignment Inspector</div>',
                unsafe_allow_html=True)

    ref_p = st.session_state.get("ref_path")
    out_p = st.session_state.get("out_path")

    if ref_p and out_p and Path(ref_p).exists() and Path(out_p).exists():
        try:
            from src.io.raster_loader import RasterLoader
            loader = RasterLoader()
            ref_arr, _ = loader.load(ref_p)
            out_arr, _ = loader.load(out_p)
            before_b64 = _arr_to_b64_png(ref_arr)
            after_b64  = _arr_to_b64_png(out_arr)
            split_curtain_widget(before_b64, after_b64)
        except Exception as exc:
            st.warning(f"Could not render split curtain: {exc}")
    else:
        st.info("Run the pipeline to see the split curtain comparison.")


# ─── Tab 2: 3D Terrain Viewer ─────────────────────────────────────────────────
with tab_3d:
    st.markdown('<div class="section-title">🏔️ Interactive 3D Lunar Terrain Viewer</div>',
                unsafe_allow_html=True)

    st.caption(
        f"Z-scale: {z_scale}× | Spatial downsampling: 1/{down_factor} | "
        "Powered by PyVista + Trame (install: `pip install pyvista trame trame-vtk`)"
    )

    terrain_html = st.session_state.get("terrain_html")
    default_html = ROOT / "data" / "processed" / "terrain_3d.html"
    terrain_path = terrain_html if terrain_html else (default_html if default_html.exists() else None)

    col_dem_l, col_dem_r = st.columns([3, 1])
    with col_dem_r:
        if st.button("🔄 Regenerate Synthetic DEM", use_container_width=True):
            from src.geometry.mesh_builder import SurfaceMeshBuilder
            builder = SurfaceMeshBuilder(z_scale=z_scale, downsample=down_factor)
            dem = builder.generate_synthetic_dem(width=256, height=256)
            out = ROOT / "data" / "processed" / "terrain_3d.html"
            builder.build_and_export(elevation=dem, output_path=out)
            st.session_state["terrain_html"] = out
            st.rerun()

        # Use registered raster as texture on terrain
        out_p = st.session_state.get("out_path")
        if out_p and Path(out_p).exists():
            if st.button("🗺️ Apply Registered Image as Texture", use_container_width=True):
                from src.io.raster_loader import RasterLoader
                from src.geometry.mesh_builder import SurfaceMeshBuilder
                loader  = RasterLoader()
                reg_arr, _ = loader.load(out_p)
                builder = SurfaceMeshBuilder(z_scale=z_scale, downsample=down_factor)
                dem = builder.generate_synthetic_dem(width=reg_arr.shape[-1],
                                                      height=reg_arr.shape[-2])
                out = ROOT / "data" / "processed" / "terrain_textured.html"
                builder.build_and_export(elevation=dem, output_path=out,
                                         texture_image=reg_arr)
                st.session_state["terrain_html"] = out
                st.rerun()

    with col_dem_l:
        terrain_3d_viewer(terrain_path, height=580)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Copyright (c) 2026 Santhosh Jayakumar & Team — MIT License | ISRO SIH26166 | Yaazhi GeoAlign OS")
