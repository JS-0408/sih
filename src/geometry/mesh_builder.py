"""
src/geometry/mesh_builder.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Copyright (c) 2026 Santhosh Jayakumar & Team — MIT License
Part of Yaazhi GeoAlign OS / ISRO SIH26166 Chandrayaan-2 Registration System.

3D Surface Mesh Builder using PyVista.

Constructs a StructuredGrid 3D terrain mesh from a 2D elevation array
(Digital Elevation Model) optionally textured with a registered raster image.
Exports the interactive scene as a self-contained HTML file for embedding
in Streamlit or browser dashboards.

Usage:
    from src.geometry.mesh_builder import SurfaceMeshBuilder
    builder = SurfaceMeshBuilder(z_scale=2.0, downsample=4)
    html_path = builder.build_and_export(
        elevation=dem_array,
        texture_image=registered_rgb,
        output_path=Path("data/processed/terrain_3d.html"),
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("MeshBuilder")


class SurfaceMeshBuilder:
    """
    Builds a 3D terrain mesh from a 2D elevation (DEM) array and
    optionally overlays a registered raster image as a surface texture.

    Parameters
    ----------
    z_scale : float
        Vertical exaggeration factor applied to the elevation values.
        Values > 1 amplify terrain relief for visualization.
    downsample : int
        Spatial downsampling factor. A value of 4 means every 4th pixel
        is sampled, reducing the mesh resolution and memory footprint.
    """

    def __init__(self, z_scale: float = 1.5, downsample: int = 4) -> None:
        self.z_scale   = z_scale
        self.downsample = max(1, int(downsample))

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def build_and_export(
        self,
        elevation: np.ndarray,
        output_path: str | Path,
        texture_image: Optional[np.ndarray] = None,
    ) -> Path:
        """
        Build a 3D terrain StructuredGrid and export it to an HTML file.

        Parameters
        ----------
        elevation : np.ndarray
            2D array of elevation values (H × W), any numeric dtype.
        output_path : str | Path
            Destination path for the exported HTML file.
        texture_image : np.ndarray or None
            Optional (H × W) or (H × W × 3) uint8 image for surface coloring.
            If None, colormap shading is applied to the elevation values instead.

        Returns
        -------
        Path
            Absolute path to the generated HTML file.
        """
        try:
            import pyvista as pv
        except ImportError:
            logger.warning(
                "PyVista not installed. Install with: pip install pyvista trame trame-vtk\n"
                "Falling back to lightweight HTML placeholder export."
            )
            return self._export_fallback_html(elevation, output_path)

        logger.info(
            f"Building 3D mesh: z_scale={self.z_scale}, downsample={self.downsample}, "
            f"input shape={elevation.shape}"
        )

        # ── Downsample ─────────────────────────────────────────────────────
        dem = elevation[:: self.downsample, :: self.downsample].astype(np.float32)
        h, w = dem.shape

        # ── Normalize elevation to [0, 1] then scale ───────────────────────
        dem_min, dem_max = dem.min(), dem.max()
        dem_range = dem_max - dem_min if dem_max > dem_min else 1.0
        dem_norm = (dem - dem_min) / dem_range  # [0, 1]

        # ── Build X, Y, Z coordinate grids ────────────────────────────────
        x = np.linspace(0, w - 1, w, dtype=np.float32)
        y = np.linspace(0, h - 1, h, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        zz = dem_norm * self.z_scale * max(h, w) * 0.05   # proportional Z

        # ── Create PyVista StructuredGrid ──────────────────────────────────
        grid = pv.StructuredGrid(xx, yy, zz)
        grid["elevation"] = dem_norm.ravel(order="C")

        # ── Apply texture image if provided ────────────────────────────────
        if texture_image is not None:
            try:
                tex_ds = texture_image[:: self.downsample, :: self.downsample]
                if tex_ds.ndim == 2:
                    tex_rgb = np.stack([tex_ds] * 3, axis=-1)
                else:
                    tex_rgb = tex_ds[:, :, :3]
                # Normalize to uint8
                if tex_rgb.dtype != np.uint8:
                    tex_rgb = ((tex_rgb - tex_rgb.min()) /
                               max(tex_rgb.max() - tex_rgb.min(), 1) * 255
                               ).astype(np.uint8)
                tex_flat = tex_rgb.reshape(-1, 3)
                grid["RGB"] = tex_flat
            except Exception as tex_exc:
                logger.warning(f"Texture overlay failed, using elevation colormap: {tex_exc}")

        # ── Export to HTML ─────────────────────────────────────────────────
        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            pl = pv.Plotter(off_screen=True)
            scalars = "RGB" if "RGB" in grid.point_data else "elevation"
            pl.add_mesh(
                grid,
                scalars=scalars,
                show_scalar_bar=(scalars == "elevation"),
                rgb=(scalars == "RGB"),
                smooth_shading=True,
            )
            pl.set_background("#0b0f19")
            pl.camera_position = "iso"
            pl.export_html(str(out_path))
            logger.info(f"3D terrain HTML exported to: {out_path}")
        except Exception as export_exc:
            logger.warning(f"PyVista HTML export failed: {export_exc}. Using fallback.")
            return self._export_fallback_html(elevation, out_path)

        return out_path

    def generate_synthetic_dem(
        self, width: int = 256, height: int = 256, seed: int = 42
    ) -> np.ndarray:
        """
        Generate a synthetic DEM array for testing when no real DEM is available.
        Simulates lunar crater fields using Gaussian bowl shapes.

        Returns
        -------
        np.ndarray : float32 elevation array of shape (height, width)
        """
        rng = np.random.default_rng(seed)
        dem = rng.normal(0.0, 0.05, (height, width)).astype(np.float32)

        for _ in range(20):
            margin_x = min(30, width  // 4)
            margin_y = min(30, height // 4)
            if margin_x >= width - margin_x or margin_y >= height - margin_y:
                continue
            cx = rng.integers(margin_x, width  - margin_x)
            cy = rng.integers(margin_y, height - margin_y)
            r  = rng.integers(10, 40)
            depth = rng.uniform(0.1, 0.5)
            yy, xx = np.ogrid[:height, :width]
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
            rim_mask  = (dist > r * 0.85) & (dist < r * 1.15)
            bowl_mask = dist < r * 0.85
            dem[rim_mask]  += float(depth) * 0.4
            dem[bowl_mask] -= float(depth) * (1.0 - dist[bowl_mask] / r)

        return dem

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _export_fallback_html(
        self, elevation: np.ndarray, output_path: str | Path
    ) -> Path:
        """
        Export a lightweight static HTML page when PyVista is not available.
        Renders an ASCII elevation heatmap as a colored grid.
        """
        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        dem = elevation[:: self.downsample, :: self.downsample].astype(np.float32)
        dem_min, dem_max = dem.min(), dem.max()
        rng = dem_max - dem_min if dem_max > dem_min else 1.0
        norm = ((dem - dem_min) / rng * 255).astype(np.uint8)

        h, w = norm.shape
        cell_size = max(2, 600 // max(w, 1))
        cells_html = ""
        for row in norm:
            for val in row:
                r = int(val * 0.4)
                g = int(val * 0.6)
                b = int(255 - val * 0.5)
                cells_html += (
                    f'<div style="width:{cell_size}px;height:{cell_size}px;'
                    f'background:rgb({r},{g},{b});display:inline-block;"></div>'
                )
            cells_html += "<br>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>3D Terrain Viewer — Yaazhi GeoAlign OS</title>
<style>
  body {{ background:#0b0f19; color:#f3f4f6; font-family:Inter,sans-serif;
         display:flex; flex-direction:column; align-items:center; padding:2rem; }}
  h1 {{ color:#06b6d4; }}
  .grid {{ line-height:0; margin-top:1rem; }}
  .info {{ color:#9ca3af; font-size:0.85rem; margin-top:1rem; }}
</style>
</head>
<body>
<h1>Lunar DEM Elevation Map</h1>
<div class="info">
  Shape: {h}&times;{w} px | Z-scale: {self.z_scale}&times; |
  Elevation range: {dem_min:.1f}&ndash;{dem_max:.1f}
  <br><em>Install pyvista for full interactive 3D: <code>pip install pyvista trame</code></em>
</div>
<div class="grid">{cells_html}</div>
</body>
</html>"""

        out_path.write_text(html, encoding="utf-8")
        logger.info(f"Fallback DEM HTML exported to: {out_path}")
        return out_path
