/* ==========================================================================
   Yaazhi GeoAlign OS — Interactive Frontend Application Logic
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  const datasetSelect = document.getElementById("dataset-select");
  const detectorSelect = document.getElementById("detector-select");
  const tileSizeInput = document.getElementById("tile-size-input");
  const tileSizeVal = document.getElementById("tile-size-val");
  const kpInput = document.getElementById("kp-input");
  const kpVal = document.getElementById("kp-val");
  const ransacInput = document.getElementById("ransac-input");
  const ransacVal = document.getElementById("ransac-val");
  const runBtn = document.getElementById("run-btn");

  const imgBefore = document.getElementById("img-before");
  const imgAfter = document.getElementById("img-after");
  const afterWrapper = document.getElementById("after-wrapper");
  const sliderHandle = document.getElementById("slider-handle");
  const cmpContainer = document.getElementById("cmp-container");
  const placeholder = document.getElementById("placeholder-overlay");

  const verdictStatus = document.getElementById("verdict-status");
  const verdictSub = document.getElementById("verdict-sub");
  const valRmse = document.getElementById("val-rmse");
  const valCoverage = document.getElementById("val-coverage");
  const valGcps = document.getElementById("val-gcps");
  const valRuntime = document.getElementById("val-runtime");
  const matrixDisplay = document.getElementById("matrix-display");
  const downloadSection = document.getElementById("download-section");
  const downloadGeoTIFF = document.getElementById("download-geotiff");

  let datasets = [];
  let isDragging = false;

  // 1. Sync Slider Values
  tileSizeInput.addEventListener("input", (e) => tileSizeVal.textContent = e.target.value);
  kpInput.addEventListener("input", (e) => kpVal.textContent = e.target.value);
  ransacInput.addEventListener("input", (e) => ransacVal.textContent = e.target.value);

  // 2. Fetch Available Datasets
  async function loadDatasets() {
    try {
      const res = await fetch("/api/datasets");
      const data = await res.json();
      datasets = data.datasets || [];

      datasetSelect.innerHTML = "";
      if (datasets.length === 0) {
        datasetSelect.innerHTML = '<option value="">No datasets found</option>';
        return;
      }

      datasets.forEach((ds) => {
        const opt = document.createElement("option");
        opt.value = ds.id;
        opt.textContent = ds.name;
        datasetSelect.appendChild(opt);
      });
    } catch (err) {
      console.error("Failed to load datasets:", err);
    }
  }

  loadDatasets();

  // 3. Interactive Split Comparison Slider Drag Logic
  function setSliderPosition(xPos) {
    const rect = cmpContainer.getBoundingClientRect();
    let posX = xPos - rect.left;
    if (posX < 0) posX = 0;
    if (posX > rect.width) posX = rect.width;

    const percentage = (posX / rect.width) * 100;
    sliderHandle.style.left = `${percentage}%`;
    afterWrapper.style.width = `${percentage}%`;

    // Ensure the underlying after image maintains full container width to avoid squeezing
    imgAfter.style.width = `${rect.width}px`;
  }

  cmpContainer.addEventListener("mousedown", (e) => {
    isDragging = true;
    setSliderPosition(e.clientX);
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    setSliderPosition(e.clientX);
  });

  window.addEventListener("mouseup", () => {
    isDragging = false;
  });

  window.addEventListener("resize", () => {
    const rect = cmpContainer.getBoundingClientRect();
    imgAfter.style.width = `${rect.width}px`;
  });

  // 4. Run Registration Pipeline via API
  runBtn.addEventListener("click", async () => {
    const selectedId = datasetSelect.value;
    const selectedDs = datasets.find((d) => d.id === selectedId);

    if (!selectedDs) {
      alert("Please select a valid scene dataset.");
      return;
    }

    runBtn.disabled = true;
    runBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing Scene...';
    placeholder.style.display = "flex";
    placeholder.innerHTML = '<i class="fa-solid fa-spinner fa-spin placeholder-icon"></i><p>Executing multi-tile GCP registration & warping...</p>';

    const payload = {
      reference: selectedDs.reference,
      target: selectedDs.target,
      detector: detectorSelect.value,
      tile_size: parseInt(tileSizeInput.value),
      max_keypoints: parseInt(kpInput.value),
      ransac_threshold: parseFloat(ransacInput.value),
    };

    try {
      const res = await fetch("/api/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Registration failed");
      }

      // Update UI with previews
      imgBefore.src = data.target_preview_url;
      imgAfter.src = data.output_preview_url;

      imgAfter.onload = () => {
        placeholder.style.display = "none";
        setSliderPosition(cmpContainer.getBoundingClientRect().width / 2);
      };

      // Update Metrics
      verdictStatus.textContent = data.status === "SUCCESS" ? "QUALITY GATE PASSED" : "FAILED";
      verdictStatus.className = `verdict-status ${data.status === "SUCCESS" ? "pass" : ""}`;
      verdictSub.textContent = `Completed in ${data.runtime_seconds}s across ${data.tiling.grid} tile grid`;

      valRmse.textContent = `${data.metrics.global_rmse_px} px`;
      valCoverage.textContent = `${(data.metrics.spatial_coverage * 100).toFixed(1)}%`;
      valGcps.textContent = `${data.features.gcp_inliers} pts`;
      valRuntime.textContent = `${data.runtime_seconds}s`;

      if (data.homography_matrix) {
        matrixDisplay.textContent = JSON.stringify(data.homography_matrix, null, 2);
      }

      // Show Download button
      const outputFilename = data.files.output.split(/[\\/]/).pop();
      downloadGeoTIFF.href = `/outputs/${outputFilename}`;
      downloadSection.style.display = "block";

    } catch (err) {
      alert(`Error running pipeline: ${err.message}`);
      placeholder.innerHTML = `<i class="fa-solid fa-triangle-exclamation placeholder-icon" style="color:#ef4444;"></i><p>Pipeline Error: ${err.message}</p>`;
    } finally {
      runBtn.disabled = false;
      runBtn.innerHTML = '<i class="fa-solid fa-play"></i> Run Registration Engine';
    }
  });
});
