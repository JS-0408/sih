/* ==========================================================================
   Yaazhi GeoAlign OS — Console logic
   --------------------------------------------------------------------------
   INTEGRATION CONTRACT
   --------------------------------------------------------------------------
   POST /api/register  (multipart/form-data)
     files[]    — 2+ image files
     detector   — "SIFT" | "ORB"
     model      — "homography" | "affine" | "similarity"
     use_deep   — "true" | "false"
   → sync JSON response (see renderResults for expected shape)

   POST /api/register-preset  (application/json)
     reference  — absolute server-side path
     target     — absolute server-side path
     detector, model, use_deep (same as above)
   → same sync JSON response shape

   GET /api/datasets  → { datasets: [{id, name, reference, target}] }
   ========================================================================== */

const ASYNC_MODE               = false;
const REGISTER_ENDPOINT        = "/api/register";
const REGISTER_PRESET_ENDPOINT = "/api/register-preset";
const DATASETS_ENDPOINT        = "/api/datasets";
const STATUS_ENDPOINT          = (jobId) => `/api/status/${jobId}`;
const RESULTS_ENDPOINT         = (jobId) => `/api/results/${jobId}`;

/* ============================== SENSOR DETECTION ========================= */

const SENSOR_DEFS = {
  ohr: { code: "ohr", name: "OHRC",       full: "Orbiter High Resolution Camera", gsd: "0.32 m" },
  tmc: { code: "tmc", name: "TMC-2",      full: "Terrain Mapping Camera-2",       gsd: "5 m"    },
  iir: { code: "iir", name: "IIRS",       full: "Imaging IR Spectrometer",        gsd: "80 m"   },
  s2:  { code: "s2",  name: "Sentinel-2", full: "Sentinel-2 MSI",                 gsd: "10 m"   },
};

function detectSensor(filename) {
  const lower = filename.toLowerCase();

  // ISRO PDS4 pattern: ch<mission>_<inst>_...
  const isroMatch = lower.match(/^ch(\d)_([a-z]{3})_/);
  if (isroMatch) {
    const mission = `Chandrayaan-${isroMatch[1]}`;
    const instCode = isroMatch[2];
    if (instCode === "ohr") return { ...SENSOR_DEFS.ohr, mission, matched: true };
    if (instCode === "tmc") return { ...SENSOR_DEFS.tmc, mission, matched: true };
    if (instCode === "iir") return { ...SENSOR_DEFS.iir, mission, matched: true };
  }

  if (/\bohrc?\b/.test(lower)) return { ...SENSOR_DEFS.ohr, mission: "Chandrayaan-2", matched: true };
  if (/\btmc2?\b/.test(lower)) return { ...SENSOR_DEFS.tmc, mission: "Chandrayaan-2", matched: true };
  if (/\biirs?\b/.test(lower)) return { ...SENSOR_DEFS.iir, mission: "Chandrayaan-2", matched: true };

  if (/^s2[ab]_msil/i.test(filename)) {
    return { ...SENSOR_DEFS.s2, mission: "Sentinel-2", matched: true };
  }

  return {
    code: "unknown", name: "Unrecognized", full: "Not matched to a known sensor pattern",
    gsd: "—", mission: "—", matched: false,
  };
}

/* ============================== STATE ==================================== */

const state = {
  files: [],        // { id, file, sensor }
  preset: null,     // { id, name, reference, target }
  running: false,
  lastResult: null,
};

let fileIdCounter = 0;

/* ============================== DOM REFS ================================= */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const dropzone      = $("#dropzone");
const dropzoneIdle  = $("#dropzone-idle");
const fileInput     = $("#file-input");
const browseBtn     = $("#browse-btn");
const manifest      = $("#manifest");
const manifestList  = $("#manifest-list");
const manifestMode  = $("#manifest-mode");
const addMoreBtn    = $("#add-more-btn");
const clearBtn      = $("#clear-btn");
const runBar        = $("#run-bar");
const runBtn        = $("#run-btn");
const progressEl    = $("#progress");
const progressStage = $("#progress-stage");
const progressFill  = $("#progress-fill");
const resultsEl     = $("#results");
const errorBanner   = $("#error-banner");
const engineStatus  = $("#engine-status");

/* ============================== THEME ===================================== */

function initTheme() {
  const saved = localStorage.getItem("geoalign-theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const theme = saved || (prefersDark ? "dark" : "light");
  applyTheme(theme);
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("geoalign-theme", theme);
  const toggle = $("#theme-toggle");
  const text = $("#theme-toggle-text");
  toggle.setAttribute("aria-pressed", theme === "dark");
  text.textContent = theme === "dark" ? "Dark" : "Light";
}

$("#theme-toggle").addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  applyTheme(current === "dark" ? "light" : "dark");
});

/* ============================== PRESET PANEL ============================== */

function injectPresetPanel() {
  const stage = $("#stage");
  const panel = document.createElement("div");
  panel.id = "preset-panel";
  panel.innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--accent-gold);box-shadow:0 0 8px var(--accent-gold);"></span>
      <span style="font-family:var(--font-mono);font-size:11px;font-weight:600;color:var(--ink-soft);letter-spacing:0.04em;text-transform:uppercase;">
        Server Presets
      </span>
    </div>
    <select id="preset-select" class="field__select" style="flex:1;min-width:220px;">
      <option value="">— select pre-loaded dataset —</option>
    </select>
    <button id="preset-run-btn" class="btn btn--primary btn--sm" type="button" disabled>
      Run preset
    </button>
  `;
  stage.insertBefore(panel, stage.firstChild);

  fetch(DATASETS_ENDPOINT)
    .then((r) => r.json())
    .then(({ datasets }) => {
      const sel = $("#preset-select");
      (datasets || []).forEach((ds) => {
        const opt = document.createElement("option");
        opt.value = ds.id;
        opt.textContent = ds.name;
        opt.dataset.reference = ds.reference;
        opt.dataset.target    = ds.target;
        sel.appendChild(opt);
      });
    })
    .catch(() => {});

  $("#preset-select").addEventListener("change", (e) => {
    const opt = e.target.selectedOptions[0];
    if (opt && opt.value) {
      state.preset = {
        id: opt.value,
        name: opt.textContent,
        reference: opt.dataset.reference,
        target:    opt.dataset.target,
      };
      $("#preset-run-btn").disabled = false;
    } else {
      state.preset = null;
      $("#preset-run-btn").disabled = true;
    }
  });

  $("#preset-run-btn").addEventListener("click", runPreset);
}

async function runPreset() {
  if (!state.preset || state.running) return;
  state.running = true;
  if (errorBanner) errorBanner.hidden = true;
  resultsEl.hidden   = true;
  $("#preset-run-btn").disabled = true;
  progressEl.hidden  = false;
  setEngineStatus("running", "Engine running");
  simulatePipelineSteps();

  try {
    const body = {
      reference: state.preset.reference,
      target:    state.preset.target,
      detector:  $("#cfg-detector") ? $("#cfg-detector").value : "SIFT",
      model:     $("#cfg-model")    ? $("#cfg-model").value    : "homography",
      use_deep:  $("#cfg-deep")     ? $("#cfg-deep").checked   : false,
    };

    const res = await fetch(REGISTER_PRESET_ENDPOINT, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(body),
    });

    let payload;
    try { payload = await res.json(); } catch { payload = {}; }

    if (!res.ok) {
      throw new Error(payload.error || `Server error ${res.status}`);
    }

    renderResults(payload);
    setEngineStatus(
      payload.status === "PASS" ? "pass" : "fail",
      payload.status === "PASS" ? "Quality gate passed" : "Quality gate failed",
    );
  } catch (err) {
    showError("Connection error", err.message || "Unknown error from backend.");
    setEngineStatus("idle", "Engine idle");
  } finally {
    state.running = false;
    $("#preset-run-btn").disabled = false;
    progressEl.hidden = true;
  }
}

/* ============================== FILE HANDLING ============================= */

function addFiles(fileList) {
  const incoming = Array.from(fileList);
  const zeroSize = incoming.filter((f) => f.size === 0);
  if (zeroSize.length > 0) {
    showError(
      "Folders detected — please drop files, not folders",
      `"${zeroSize.map((f) => f.name).join('", "')}" has size 0. Open the folder in Explorer and drag the .img / .tif files directly.`
    );
    return;
  }
  incoming.forEach((file) => {
    const sensor = detectSensor(file.name);
    state.files.push({ id: ++fileIdCounter, file, sensor });
  });
  renderManifest();
}

function removeFile(id) {
  state.files = state.files.filter((f) => f.id !== id);
  renderManifest();
}

function clearFiles() {
  state.files = [];
  renderManifest();
  resultsEl.hidden   = true;
  if (errorBanner) errorBanner.hidden = true;
}

function renderManifest() {
  const hasFiles = state.files.length > 0;
  manifest.hidden = !hasFiles;
  dropzone.classList.toggle("has-files", hasFiles);
  dropzoneIdle.hidden = hasFiles;
  runBar.hidden = state.files.length < 2;

  const countTag = $("#manifest-count");
  if (countTag) countTag.textContent = `${state.files.length} file${state.files.length === 1 ? "" : "s"}`;

  manifestList.innerHTML = "";
  state.files.forEach((entry, idx) => {
    const li = document.createElement("li");
    li.className = "manifest__item" + (entry.sensor.matched ? "" : " is-unknown");

    const role = idx === 0 ? "reference" : "target";

    li.innerHTML = `
      <span class="manifest__badge" data-sensor="${entry.sensor.code}">${sensorInitials(entry.sensor)}</span>
      <span class="manifest__info">
        <span class="manifest__filename">${escapeHtml(entry.file.name)}</span>
        <span class="manifest__sensor-name">${entry.sensor.matched ? `${entry.sensor.name} · ${entry.sensor.mission}` : "Unrecognized filename — assign sensor manually in backend config"}</span>
      </span>
      <span class="manifest__role" data-role="${role}">${role}</span>
      <span class="manifest__size">${formatBytes(entry.file.size)}</span>
      <button class="manifest__remove" type="button" aria-label="Remove ${escapeHtml(entry.file.name)}" data-remove="${entry.id}">×</button>
    `;
    manifestList.appendChild(li);
  });

  $$("[data-remove]").forEach((btn) => {
    btn.addEventListener("click", () => removeFile(Number(btn.dataset.remove)));
  });

  updateManifestMode();
  updateSensorChips();
}

function sensorInitials(sensor) {
  if (sensor.code === "unknown") return "?";
  if (sensor.code === "s2") return "S2";
  return sensor.name.split("-")[0].slice(0, 3).toUpperCase();
}

function updateManifestMode() {
  const sensors = new Set(state.files.map((f) => f.sensor.code));
  if (sensors.size === 0) { manifestMode.textContent = "—"; return; }
  if (sensors.has("unknown")) { manifestMode.textContent = "Needs sensor assignment"; return; }
  manifestMode.textContent = sensors.size > 1 ? "Multi-modal registration" : "Same-sensor registration";
}

function updateSensorChips() {
  const active = new Set(state.files.map((f) => f.sensor.code));
  $$(".sensor-chip").forEach((chip) => {
    chip.classList.toggle("is-active", active.has(chip.dataset.sensor));
  });
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

/* ---- drag & drop / browse wiring ---- */

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
addMoreBtn.addEventListener("click", () => fileInput.click());
clearBtn.addEventListener("click", clearFiles);

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) addFiles(e.target.files);
  fileInput.value = "";
});

["dragenter", "dragover"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("is-dragover");
  });
});
["dragleave", "drop"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("is-dragover");
  });
});
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
});

/* ============================== RUN ORCHESTRATION ========================= */

const PIPELINE_STEPS = ["io", "tiling", "features", "matching", "ransac", "gcp", "warp", "quality"];

runBtn.addEventListener("click", runRegistration);

async function runRegistration() {
  if (state.running || state.files.length < 2) return;
  state.running = true;
  if (errorBanner) errorBanner.hidden = true;
  resultsEl.hidden   = true;
  runBtn.disabled    = true;
  runBtn.classList.add("is-loading");
  progressEl.hidden = false;
  setEngineStatus("running", "Engine running");
  simulatePipelineSteps();

  try {
    const formData = new FormData();
    state.files.forEach((entry) => formData.append("files[]", entry.file, entry.file.name));
    formData.append("detector", $("#cfg-detector").value);
    formData.append("model",    $("#cfg-model").value);
    formData.append("use_deep", $("#cfg-deep").checked ? "true" : "false");

    const res = await fetch(REGISTER_ENDPOINT, { method: "POST", body: formData });

    let payload;
    try { payload = await res.json(); } catch { payload = {}; }

    if (!res.ok) {
      throw new Error(payload.error || `Server error ${res.status}`);
    }

    if (ASYNC_MODE && payload.job_id) {
      payload = await pollJob(payload.job_id);
    }

    renderResults(payload);
    setEngineStatus(
      payload.status === "PASS" ? "pass" : "fail",
      payload.status === "PASS" ? "Quality gate passed" : "Quality gate failed",
    );
  } catch (err) {
    const msg = err.message === "Failed to fetch"
      ? "Could not reach the backend server. Make sure server.py is running on port 5000."
      : err.message;
    showError("Server Connection Error", msg);
    setEngineStatus("idle", "Engine idle");
  } finally {
    state.running = false;
    runBtn.disabled = false;
    runBtn.classList.remove("is-loading");
    progressEl.hidden = true;
  }
}

async function pollJob(jobId) {
  for (let i = 0; i < 240; i++) {
    await sleep(1000);
    const statusRes = await fetch(STATUS_ENDPOINT(jobId));
    if (statusRes.ok) {
      const s = await statusRes.json();
      if (s.stage) setActivePipelineStep(s.stage);
      if (typeof s.progress_pct === "number") progressFill.style.width = `${s.progress_pct}%`;
      if (s.stage === "done" || s.status) break;
    }
  }
  const resultsRes = await fetch(RESULTS_ENDPOINT(jobId));
  if (!resultsRes.ok) throw new Error(`Results fetch failed (${resultsRes.status})`);
  return resultsRes.json();
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

let simTimer = null;
function simulatePipelineSteps() {
  let i = 0;
  progressFill.style.width = "4%";
  clearInterval(simTimer);
  simTimer = setInterval(() => {
    if (i >= PIPELINE_STEPS.length || !state.running) { clearInterval(simTimer); return; }
    setActivePipelineStep(PIPELINE_STEPS[i]);
    progressFill.style.width = `${Math.min(96, ((i + 1) / PIPELINE_STEPS.length) * 100)}%`;
    i++;
  }, 650);
}

const STEP_LABELS = {
  io:       "Reading raster headers…",
  tiling:   "Building spatial tile grid…",
  features: "Detecting keypoints…",
  matching: "Matching features (FLANN)…",
  ransac:   "Rejecting outliers (RANSAC)…",
  gcp:      "Aggregating ground control points…",
  warp:     "Applying sub-pixel warp…",
  quality:  "Running quality gate…",
};

function setActivePipelineStep(step) {
  progressStage.textContent = STEP_LABELS[step] || "Processing…";
  $$("#progress-pipeline span").forEach((el) => {
    const idx = PIPELINE_STEPS.indexOf(el.dataset.step);
    const activeIdx = PIPELINE_STEPS.indexOf(step);
    el.classList.toggle("is-active", el.dataset.step === step);
    el.classList.toggle("is-done",   idx < activeIdx);
  });
}

function setEngineStatus(state_, label) {
  engineStatus.dataset.state = state_;
  engineStatus.querySelector(".status-pill__label").textContent = label;
}

/* ============================== RESULTS RENDERING ========================= */

function renderResults(data) {
  state.lastResult = data;
  clearInterval(simTimer);
  progressFill.style.width = "100%";
  resultsEl.hidden = false;
  if (errorBanner) errorBanner.hidden = true; // Ensure red error banner stays hidden

  const pass = String(data.status).toUpperCase() === "PASS";
  const banner = $("#results-banner");
  banner.dataset.status = pass ? "pass" : "fail";
  $("#results-banner-icon").textContent = pass ? "✓" : "✕";

  if (pass) {
    $("#results-banner-title").textContent = "Quality Gate Passed";
    $("#results-banner-sub").textContent = `Alignment verified cleanly. Sub-pixel RMSE (${data.rmse} px) and spatial coverage (${data.coverage_pct}%) pass all thresholds.`;
  } else {
    // Domain-specific clear explanations for non-passing Quality Gate results
    if (data.inliers < (data.min_inliers || 4)) {
      $("#results-banner-title").textContent = "No Common Feature Correspondence Found";
      $("#results-banner-sub").textContent = "The uploaded images do not share sufficient common lunar surface terrain or crater landmarks. Verify that both frames cover the same geographic region.";
    } else if (data.coverage_pct < 15) {
      $("#results-banner-title").textContent = `Spatial Coverage Limited (${data.coverage_pct}%)`;
      $("#results-banner-sub").textContent = `Feature correspondences were detected in a localized region (${data.coverage_pct}% vs 15.0% required). Alignment across the entire frame is unverified.`;
    } else {
      $("#results-banner-title").textContent = "Quality Gate Warning";
      $("#results-banner-sub").textContent = "High residual alignment error detected across control points. Try selecting an Affine transform model or SuperPoint deep matcher.";
    }
  }

  setMetric("rmse",     data.rmse,        `${data.rmse} / ${data.max_rmse ?? "—"} px max`, data.max_rmse ? Math.min(100, (data.rmse / data.max_rmse) * 100) : 0);
  setMetric("coverage", data.coverage_pct, `${data.coverage_pct}% of scene`,                data.coverage_pct);
  setMetric("inliers",  data.inliers,      `min ${data.min_inliers ?? "—"} required`,       data.min_inliers ? Math.min(100, (data.inliers / (data.min_inliers * 20)) * 100) : 0);
  setMetric("time",     data.elapsed_s,    "wall-clock",                                    Math.min(100, (data.elapsed_s / 30) * 100));

  $("#dt-sensor-pair").textContent = data.sensor_pair || sensorPairFromState();
  $("#dt-mode").textContent        = manifestMode.textContent;
  $("#dt-detector").textContent    = data.detector || "—";
  $("#dt-model").textContent       = data.model    || "—";
  $("#dt-tiles").textContent       = data.tile_grid ? `${data.tile_grid.rows} × ${data.tile_grid.cols}` : "—";
  $("#dt-keypoints").textContent   = (data.keypoints_ref != null && data.keypoints_tgt != null)
    ? `${data.keypoints_ref} / ${data.keypoints_tgt}` : "—";
  $("#dt-ratio").textContent       = data.match_ratio  ?? "—";
  $("#dt-min-inliers").textContent = data.min_inliers  ?? "—";
  $("#dt-max-rmse").textContent    = data.max_rmse ? `${data.max_rmse} px` : "—";
  $("#dt-resolution").textContent  = data.output_resolution || "—";

  renderTileGrid(data.tile_grid, data.tile_inlier_density);
  renderPreview(data.preview_ref_url, data.preview_reg_url);
  renderLog(data.log);

  resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });
}

function setMetric(key, value, subtitle, barPct) {
  const valueEl = $(`#metric-${key}`);
  valueEl.textContent = value ?? "—";
  const bar = $(`#metric-${key}-bar`);
  if (bar) bar.style.width = `${Math.max(0, Math.min(100, barPct || 0))}%`;
}

function sensorPairFromState() {
  const names = [...new Set(state.files.map((f) => f.sensor.name))];
  return names.join(" → ");
}

function renderTileGrid(tileGrid, density) {
  const grid = $("#tile-grid");
  grid.innerHTML = "";
  if (!tileGrid || !density) {
    grid.style.gridTemplateColumns = "repeat(12, 1fr)";
    for (let i = 0; i < 24; i++) {
      const t = document.createElement("div");
      t.className = "tile";
      grid.appendChild(t);
    }
    return;
  }
  grid.style.gridTemplateColumns = `repeat(${tileGrid.cols}, 1fr)`;
  density.forEach((v) => {
    const t = document.createElement("div");
    t.className = "tile";
    const clamped = Math.max(0, Math.min(1, v));
    t.style.background = `color-mix(in srgb, var(--primary) ${Math.round(clamped * 100)}%, var(--surface-sunk))`;
    t.title = `${Math.round(clamped * 100)}% inlier density`;
    grid.appendChild(t);
  });
}

function renderPreview(refUrl, regUrl) {
  const placeholder = $("#preview-placeholder");
  const refImg      = $("#preview-ref");
  const regImg      = $("#preview-reg");

  if (!refUrl && !regUrl) {
    placeholder.hidden = false;
    refImg.hidden = true;
    regImg.hidden = true;
    return;
  }
  placeholder.hidden = true;

  const ts = Date.now();
  if (refUrl) {
    refImg.src = `${refUrl}${refUrl.includes('?') ? '&' : '?'}t=${ts}`;
    refImg.hidden = false;
  }
  if (regUrl) {
    regImg.src = `${regUrl}${regUrl.includes('?') ? '&' : '?'}t=${ts}`;
    regImg.hidden = false;
  }

  setPreviewView("overlay");
}

function setPreviewView(mode) {
  const container = $("#preview-container");
  if (container) container.dataset.view = mode;
  $$(".preview-toggle__btn").forEach((b) => {
    b.classList.toggle("is-active", b.dataset.view === mode);
  });
}

$$(".preview-toggle__btn").forEach((btn) => {
  btn.addEventListener("click", () => setPreviewView(btn.dataset.view));
});

function renderLog(lines) {
  const log = $("#log-output");
  if (!lines || !lines.length) {
    log.textContent = "No pipeline log returned by the backend for this run.";
    return;
  }
  log.textContent = lines.map((l, i) => `[${String(i + 1).padStart(2, "0")}] ${l}`).join("\n");
}

$("#copy-log-btn").addEventListener("click", () => {
  navigator.clipboard?.writeText($("#log-output").textContent);
});

/* ============================== ERROR STATE ================================ */

function showError(title, detail) {
  alert(`${title}\n\n${detail}`);
}

/* ============================== REPORT DOWNLOAD ============================= */

$("#download-report-btn").addEventListener("click", () => {
  if (!state.lastResult) return;
  const blob = new Blob([JSON.stringify(state.lastResult, null, 2)], { type: "application/json" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url;
  a.download = "geoalign_summary.json";
  a.click();
  URL.revokeObjectURL(url);
});

/* ============================== INIT ========================================= */

if (errorBanner) errorBanner.hidden = true;
initTheme();
renderManifest();
injectPresetPanel();
