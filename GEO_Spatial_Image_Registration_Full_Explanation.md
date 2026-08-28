## GEO SPATIAL IMAGE REGISTRATION

Full Technical Explanation, Architecture, Algorithms, Memory Strategy, Distributed Execution,

Deep Matching, Validation, and Production Workflow

| Document scope | End-to-end reference for a memory-safe geospatial image registration pipeline |
| --- | --- |
| Primary imagery | Large GeoTIFF satellite / planetary / lunar raster scenes |
| Core baseline | Windowed RasterIO + SIFT + grid filtering + FLANN + Lowe ratio test + RANSAC |
| Scaling layer | Ray distributed tile execution using lightweight task arguments |
| Deep matching layer | SuperPoint + LightGlue / optional LoFTR, inference-only and low-memory operation |
| Outputs | Registered GeoTIFF, transformation metadata, control-point metrics, RMSE, and summary JSON |
| Primary validation tool | QGIS visual inspection plus quantitative geometric metrics |

Executive summary. The project solves a practical remote-sensing problem: imagery of the same geographic or planetary scene acquired at different times, viewing geometries, illumination conditions, or sensors does not necessarily align pixel-for-pixel. The pipeline therefore combines memory-safe raster access, spatially balanced feature extraction, robust descriptor matching, geometric outlier rejection, geospatially correct warping, distributed execution, and optional deep feature matching.

The most important engineering principle is to separate data access, feature extraction, matching, geometric estimation, warping, and evaluation. This makes each stage testable and prevents memory or geometric errors from being hidden inside a single monolithic program.

## TABLE OF CONTENTS

- 1. Problem Statement and Scientific Motivation

- 2. System Goals and Non-Goals

- 3. End-to-End Architecture

- 4. Phase 1 — Memory-Safe Raster Processing and Classical Registration

- 5. Phase 2 — Distributed Ray Execution

- 6. Phase 3 — Deep Feature Matching

- 7. Phase 4 — Production Orchestration and Quality Control

- 8. Geometric Models and Transformation Strategy

- 9. Memory and Hardware Engineering

- 10. Configuration Design

- 11. Repository and Module Responsibilities

- 12. Metrics, Validation, and Failure Detection

- 13. QGIS and GeoTIFF Output Requirements

- 14. Technical Hurdles and Solutions

- 15. Benchmarking and Experimental Plan

- 16. Recommended Implementation Sequence

- 17. Risks, Blind Spots, and Corrections

- 18. Final End-to-End Workflow


## 1. Problem Statement and Scientific Motivation

Image registration is the process of estimating the spatial relationship between two or more images of the same physical scene so that corresponding locations occupy consistent coordinates. In remote sensing, this is essential before operations such as change detection, temporal analysis, mosaicking, elevation inference, terrain comparison, or multi-sensor fusion.

Satellite and lunar imagery can differ because of orbital geometry, sensor viewpoint, camera characteristics, terrain relief, acquisition time, seasonal conditions, atmospheric effects for Earth observations, and especially illumination/shadow changes over planetary terrain.

A naive implementation often fails before registration even starts: a very large GeoTIFF can occupy multiple gigabytes after decompression. Reading an entire raster into a NumPy array creates a peak-memory requirement that can exceed the available RAM on an 8 GB laptop. The design therefore treats the raster as a disk-backed dataset and processes bounded windows.

| Challenge | Why it matters | Pipeline response |
| --- | --- | --- |
| Very large GeoTIFFs | Full-array loading can exceed RAM | Windowed RasterIO reads with bounded tile dimensions |
| Weak or repetitive texture | Few distinctive correspondences | Spatial grid filtering plus stronger matching methods |
| Illumination and shadow | Creates false or unstable matches | Ratio test, geometric verification, robust RANSAC |
| changes |   |   |
| Local terrain distortion | One global transform may be inadequate | Evaluate affine, homography, and potentially local models |
| Distributed memory | Large Ray objects can exhaust shared memory Pass file paths and window coordinates, not image arrays |   |
| pressure |   |   |
| Deep-model memory | FP32/autograd consume unnecessary memory | Inference-only execution and mixed/half precision where |
| spikes |   | supported |
| Geospatial metadata loss | Output may no longer align in GIS | Preserve CRS, transform, resolution, bounds, and raster |
|   |   | profile |

Key principle: a visually plausible alignment is not enough. The system must produce an alignment that is both geometrically defensible and geospatially interpretable.

## 2. System Goals and Non-Goals

| Goals | Non-goals / boundaries |
| --- | --- |
| Process multi-gigabyte rasters without requiring full-raster RAM. | Guaranteeing perfect registration for arbitrary terrain and |
|   | acquisition conditions. |
| Produce repeatable feature correspondences and geometric | Assuming one transformation model works for every sensor and |
| transforms. | terrain. |
| Scale tile-level work across available laptops. | Passing raw multi-megabyte image matrices through Ray when |
|   | avoidable. |
| Support classical and deep matching paths. | Using deep learning simply because it is newer; it must beat the |
|   | baseline on relevant cases. |
| Preserve geospatial semantics in outputs. | Treating an affine metadata update as equivalent to a full pixel |
|   | warp. |
| Generate quantitative quality metrics. | Relying solely on console logs or visual inspection. |

## 3. End-to-End Architecture

## Logical pipeline:

Reference GeoTIFF + Target GeoTIFF → tile/window generation → preprocessing → feature extraction → spatial grid selection → descriptor matching → ratio filtering → geometric verification → transformation estimation → control-point aggregation → raster warping/resampling → geospatial metadata preservation → metrics → registered GeoTIFF + summary.json


| Layer | Responsibility | Primary technologies |
| --- | --- | --- |
| Input / I/O | Read bounded raster windows and expose geospatial | RasterIO, Window, NumPy |
|   | metadata |   |
| Processing | Normalize imagery and enforce spatial feature distribution | OpenCV, NumPy |
| Classical matching | Generate correspondences and reject outliers | SIFT, FLANN, Lowe ratio, RANSAC |
| Deep matching | Handle difficult low-texture or illumination cases | PyTorch, SuperPoint, LightGlue / LoFTR |
| Distributed execution Run independent tile jobs without moving large arrays |   | Ray |
| Geometry | Estimate and validate spatial transformations | OpenCV / robust estimation |
| Output | Warp raster and preserve GIS metadata | RasterIO |
| Evaluation | Measure inliers, RMSE, coverage, runtime and failure rate | Python / JSON |
| Visualization | Inspect overlays and residual misalignment | QGIS |

A useful mental model is that pixels stay close to storage. The distributed scheduler moves instructions and small metadata rather than moving enormous image arrays between machines.

## 4. Phase 1 — Memory-Safe Raster Processing and Classical Registration

Phase 1 establishes the scientifically interpretable baseline. It should be implemented and benchmarked before distributed or deep-learning complexity is introduced.

## 4.1 Windowed Raster Streaming

The raster loader uses RasterIO's windowed access pattern. Instead of reading the full raster, the loader requests a rectangular Window such as 1024×1024 pixels. Adjacent windows can overlap by a configured fraction, for example 10%, so features close to a tile boundary are less likely to be lost.

| Parameter | Example | Engineering purpose |
| --- | --- | --- |
| Tile width / height | 1024 × 1024 | Controls peak per-task memory and feature workload |
| Overlap | 10% | Reduces boundary effects and lost correspondences |
| Read dtype | uint8 / float32 as needed | Avoids accidental promotion to larger types |
| Bands | One grayscale band or | Reduces unnecessary memory |
|   | selected band |   |
| Window scheduling | Deterministic row/column | Improves reproducibility |
|   | order |   |

The loader should return both the pixel array and enough spatial context to reconstruct its geographic location: window offsets, source transform, CRS, pixel size, and relevant bounds. The array lifetime should end as soon as downstream computation is complete.

Important correction: gc.collect() is a cleanup aid, not the primary memory guarantee. Python garbage collection does not force every allocator to return memory to the operating system. The actual memory guard is bounded tile size × bounded concurrency × bounded model state.

## 4.2 Preprocessing

- Convert the selected band(s) to a representation appropriate for feature extraction.

- Normalize intensity when acquisition conditions differ substantially, while retaining the option to compare raw and normalized paths.

- Avoid expensive full-scene preprocessing. Apply it per window or via streaming operations.

- Record preprocessing settings in the configuration and summary output so experiments are reproducible.


## 4.3 SIFT Feature Extraction and Grid Filtering

SIFT provides keypoints and descriptors that are comparatively robust to scale and rotation changes. However, unrestricted extraction can concentrate keypoints in a few high-contrast regions. The proposed grid strategy divides each tile into spatial cells, such as 100×100 pixels, then retains the strongest N keypoints per cell.

| Stage | Operation | Expected effect |
| --- | --- | --- |
| Detect | Run SIFT on the tile | Generate candidate keypoints/descriptors |
| Assign | Map each keypoint to a grid cell | Expose spatial concentration |
| Rank | Sort candidates by response strength | Prefer stable/high-contrast candidates |
| Cap | Keep at most N per cell | Prevent one region from dominating |
| Merge | Concatenate retained descriptors | Create spatially balanced tile representation |

The grid does not create information that does not exist. In extremely texture-poor regions, it may leave cells nearly empty. Its purpose is to prevent the opposite failure: having enough features overall but having all of them in one small portion of the image.

## 4.4 FLANN + Lowe Ratio Test

For each descriptor in image A, the matcher searches for nearest descriptors in image B. Lowe's ratio test compares the nearest distance d1 with the second-nearest distance d2 and accepts the candidate when d1/d2 is

below a threshold such as 0.75.

The ratio threshold should be configurable and experimentally evaluated. A strict threshold generally reduces false matches at the cost of recall; a relaxed threshold increases candidate correspondences but transfers more work to geometric verification.

## 4.5 RANSAC Geometric Verification

RANSAC estimates a geometric transformation from tentative correspondences while rejecting outliers. The critical output is not merely the matrix: it is the set of inlier correspondences, the residual error distribution, and whether the estimated model is physically plausible.

| Metric | Meaning | Warning sign |
| --- | --- | --- |
| Raw matches | Correspondences before geometric filtering | Very low count may indicate feature failure |
| Inliers | Matches consistent with estimated geometry | Very low count indicates unreliable alignment |
| Inlier ratio | Inliers / raw matches | Very low ratio suggests many false matches |
| Residual / RMSE | Geometric discrepancy of inliers | Large value suggests poor model or wrong |
|   |   | scale |
| Spatial coverage | Area/extent occupied by inliers | Clustered points can make model unstable |

RANSAC threshold must be interpreted in the coordinate units of the points being estimated, usually pixels for image-space estimation. It should not be chosen blindly; it must correspond to expected localization noise and image resolution.

## 4.6 Transformation Model Selection

| Model | Degrees of | Strength | Failure mode |
| --- | --- | --- | --- |
|   | freedom |   |   |
| Translation | 2 | Very stable for simple shifts | Cannot model rotation/scale/shear |
| Similarity | 4 | Handles translation, rotation, uniform scale Cannot model shear/projective effects |   |
| Affine | 6 | Handles translation, rotation, scale, shear | Cannot model projective distortion |
| Homography | 8 effective | Handles planar projective changes | Can overfit weak/collinear |
|   | parameters |   | correspondences |
| Local / piecewise model Many |   | Can represent spatially varying distortion | Needs dense, reliable control points and |
|   |   |   | careful regularization |


The project should begin with affine estimation when justified, but the correct model depends on acquisition geometry. For non-planar terrain viewed from different angles, relief displacement can create spatially varying error that no single affine transform or homography can fully eliminate.

## 4.7 GeoTIFF Warping and Export

OpenCV image warping operates primarily in pixel coordinates and does not automatically preserve the complete geospatial meaning of a GeoTIFF. The output writer therefore needs to coordinate pixel-space transformation with the source CRS, transform, resolution, dimensions, and geographic extent.

Critical distinction: changing the GeoTIFF affine transform changes the mapping from raster indices to geographic coordinates; it does not, by itself, resample pixels according to an arbitrary image-space homography. A correct implementation must decide whether the registration is represented as a new raster grid, a geospatial transform, or both, and then apply the corresponding resampling.

## 5. Phase 2 — Distributed Ray Execution

Phase 2 distributes independent or semi-independent tile jobs across two laptops connected to a local network. The central design constraint is to keep Ray object payloads small.

## 5.1 Ray Task Design

A remote task should receive values such as reference_path, target_path, window coordinates, algorithm parameters, and perhaps a small identifier. The worker then opens the raster locally and reads only its assigned

windows.

| Bad pattern | Why it fails | Preferred pattern |
| --- | --- | --- |
| ray.remote(image_array) | Large object enters Ray's | ray.remote(reference_path, target_path, |
|   | serialization/shared-memory path | window) |
| Return full tile arrays | Creates unnecessary object-store traffic | Return compact |
|   |   | matches/metrics/transform |
| Unbounded worker count | CPU/RAM contention can freeze the machine | Use explicit num_cpus and bounded |
|   |   | concurrency |
| Duplicate large model per worker | VRAM/RAM exhaustion | Control model lifetime and worker count |

## 5.2 Resource Throttling

Using a constraint such as @ray.remote(num_cpus=2) communicates the expected CPU budget to Ray. The exact value should be benchmarked rather than assumed. A machine with 8 logical cores may perform worse when all cores are saturated because OpenCV, BLAS, PyTorch, and the OS may each create their own threads.

The system should also consider internal thread counts. Otherwise, two Ray workers can each launch multiple native threads, creating hidden oversubscription.

## 5.3 Dual-Laptop Execution Model

| Component | Responsibility |
| --- | --- |
| Head / driver | Reads configuration, creates tile task list, dispatches jobs, aggregates results |
| Worker A | Opens source files available through the configured filesystem/network path and processes assigned |
|   | windows |
| Worker B | Processes another subset of windows |
| Shared storage | Provides consistent access to the reference and target GeoTIFFs |
| Result aggregation | Collects compact transformations, control points, metrics and failure records |

A major operational requirement is filesystem accessibility. If the laptops do not share the same path namespace or files, a worker receiving a path string may fail even though the driver can read the file. The deployment design must therefore specify shared storage, synchronized local copies, or a data staging


mechanism.

## 6. Phase 3 — Deep Feature Matching

Deep matching is an enhancement path for scenes where SIFT produces too few reliable correspondences or where illumination and texture changes make handcrafted descriptors inadequate.

## 6.1 SuperPoint + LightGlue

The conceptual flow is: tile pair → SuperPoint keypoint/descriptor extraction → LightGlue learned matching → geometric verification → transformation estimation. SuperPoint supplies learned local features, while LightGlue estimates correspondences using a learned matching mechanism.

The deep model should not bypass geometric verification. Learned matches can still contain outliers, especially under severe appearance changes or repeated structures.

## 6.2 Memory Controls

| Control | Purpose | Implementation note |
| --- | --- | --- |
| torch.inference_mode() | Disable autograd bookkeeping | Use around inference-only execution |
| FP16 / autocast | Reduce activation and tensor memory where supported | Verify model/operator compatibility |
| Small tiles | Bound peak activation memory | Tune against available VRAM |
| One/few model instances | Avoid duplicate weights | Worker lifetime matters |
| Explicit device | Prevent accidental CPU/GPU duplication | Record device in metrics |
| No retained tensors | Prevent hidden memory growth | Return compact NumPy/CPU results |

FP16 should be described as a memory optimization, not a guaranteed 50% reduction in total peak memory. Some operations remain FP32, and framework/model implementation details affect actual savings.

## 6.3 LoFTR Alternative

LoFTR can be considered when detector-free dense correspondence is advantageous. It may be more demanding in memory and compute, so it should be evaluated as an alternative experiment rather than automatically inserted into every production run.

## 7. Phase 4 — Production Orchestration and Quality Control

Phase 4 turns the experimental pipeline into a reproducible command-line application.

## 7.1 Central Configuration

| Configuration group | Representative parameters |
| --- | --- |
| Raster | Input paths, bands, tile width, tile height, overlap, output path |
| Features | SIFT parameters, grid dimensions, max features per cell |
| Matching | FLANN settings, ratio threshold, deep matcher selection |
| Geometry | Model type, RANSAC threshold, confidence, iteration limit |
| Ray | Address, CPU allocation, concurrency limits |
| Deep inference | Device, precision, model weights, batch/tile limits |
| Output | Compression, dtype, resampling, metadata policy |
| Evaluation | RMSE threshold, minimum inliers, coverage threshold |

## 7.2 CLI Contract


The main entry point should support a command such as python main.py --config config/pipeline_config.yaml. The program should validate the configuration before expensive work begins, report a run identifier, and fail clearly when paths, CRS, dimensions, or model weights are invalid.

## 7.3 Summary JSON

| Metric family | Examples |
| --- | --- |
| Runtime | Total duration, per-phase duration, per-tile duration |
| Workload | Number of tiles, successful tiles, failed tiles |
| Features | Keypoints before/after grid filtering |
| Matching | Raw matches, ratio-test matches, RANSAC inliers |
| Geometry | Model type, parameters, residual RMSE, max residual |
| Coverage | Spatial extent of control points, tile coverage |
| Resources | CPU allocation, device, peak-memory observations if available |
| Output | Registered path, CRS, dimensions, resolution, transform |
| Failures | Tile ID, exception class, stage, retry count |

## 8. Geometric Models and Transformation Strategy

The transformation layer is the scientific heart of registration. It converts correspondence evidence into a spatial relationship. The system should preserve enough information to diagnose when that relationship is invalid.

## 8.1 Global vs Local Registration

A global transformation assumes one mathematical mapping explains the whole scene. This is reasonable when images differ mainly by camera pose or simple image-plane geometry. Terrain relief, parallax, orthorectification differences, or sensor-specific distortions can violate that assumption.

If residuals vary systematically with location, the system should not hide the problem by increasing the RANSAC threshold. Instead, it should diagnose whether the scene needs better orthorectification, a different transformation model, local warping, or additional control points.

## 8.2 Control-Point Aggregation

Tile-wise registration produces local correspondences. The production pipeline needs an explicit aggregation policy. Options include: combine all high-quality control points and estimate one global model; estimate a model per region; use a robust global model followed by local residual correction; or generate a piecewise

transformation field.

| Strategy | Advantage | Risk |
| --- | --- | --- |
| Global model | Simple, fast, easy to interpret | Fails under strong spatially varying distortion |
| Per-tile model | Captures local variation | Tile seams and unstable models |
| Global + local correction | Balances stability and flexibility | More implementation complexity |
| Piecewise mesh | Can model complex distortions | Needs dense, well-distributed control points |

## 9. Memory and Hardware Engineering

Memory safety should be designed quantitatively. A useful approximate budget is:

Peak memory ≈ concurrent tiles × tile bytes × active intermediate buffers + model memory + framework

overhead + OS headroom.

For an 8 GB machine, leaving substantial headroom is safer than targeting 8 GB exactly. Multiple processes, native libraries, Ray, filesystem caching, and the operating system all consume memory.


| Risk | Typical cause | Mitigation |
| --- | --- | --- |
| RAM OOM | Full raster or oversized tiles | Windowed reads and smaller tiles |
| Shared-memory exhaustion | Large Ray object payloads | Paths + windows; compact returns |
| CPU saturation | Too many Ray workers / native threads | num_cpus limits and thread control |
| GPU OOM | Large deep-model tiles / many workers | Smaller tiles, FP16, limited model workers |
| Memory fragmentation | Repeated large allocations | Reuse buffers where practical; bound concurrency |
| Disk I/O bottleneck | Many workers reading same huge files | Benchmark local vs shared storage; schedule |
|   |   | intelligently |

The pipeline is therefore not purely CPU-bound or GPU-bound. Large-scale registration can become I/O-bound, serialization-bound, or synchronization-bound. Benchmark each layer separately.

## 10. Configuration Design

A single YAML file should be the experiment contract. Every parameter affecting reproducibility should live there or be recorded automatically.

| Section | Suggested keys |
| --- | --- |
| input | reference_path, target_path, band, nodata |
| tiling | width, height, overlap |
| preprocessing | normalize, clip_percentiles, scale |
| sift | nfeatures, contrastThreshold, edgeThreshold, sigma |
| grid | cell_width, cell_height, max_per_cell |
| matching | method, ratio_threshold, checks |
| ransac | model, reprojection_threshold, confidence, max_iters |
| deep | enabled, model, weights, device, precision |
| ray | address, num_cpus, max_concurrency |
| output | path, dtype, compression, resampling |
| evaluation | min_inliers, max_rmse, min_coverage |

Configuration values should be validated for contradictions, such as a tile overlap larger than the tile dimensions, invalid RANSAC thresholds, missing model weights, or GPU selection on a machine without a compatible device.

## 11. Repository and Module Responsibilities

| File | Responsibility | Should not own |
| --- | --- | --- |
| src/io/raster_loader.py | Windowed reads, raster metadata, tile generation | Matching logic |
| src/processing/sift_grid.py | SIFT extraction and spatial bucketing | GeoTIFF writing |
| src/matching/flann_ransac.py | Descriptor matching and robust geometry | Ray scheduling |
| src/matching/deep_matcher.py | SuperPoint/LightGlue/LoFTR inference | Global orchestration |
| src/io/raster_writer.py | Geospatially correct raster export | Feature extraction |
| src/metrics/evaluator.py | RMSE, inliers, coverage, quality checks | Tile I/O |
| scripts/ray_dispatcher.py | Task dispatch and resource constraints | Image algorithms |
| main.py | CLI, pipeline orchestration, logging, result aggregation | Low-level matching implementation |
| config/pipeline_config.yaml | Experiment parameters | Business logic |


## 12. Metrics, Validation, and Failure Detection

A registration pipeline should reject bad outputs rather than merely generate them.

| Metric | Formula / concept | Interpretation |
| --- | --- | --- |
| Inlier ratio | inliers / verified matches | Higher generally indicates cleaner geometry |
| RMSE | sqrt(mean(residual²)) | Average geometric error of accepted points |
| Median residual | median(|residual|) | Less sensitive to a few large errors |
| Maximum residual | max(|residual|) | Detects severe local failures |
| Spatial coverage | extent / convex hull of control points | Checks whether geometry is supported across the scene |
| Tile success rate | successful tiles / attempted tiles | Measures operational robustness |
| Runtime/tile | tile processing time | Finds pathological regions |

RMSE alone is insufficient. Ten control points clustered in one corner can yield an excellent local RMSE while providing almost no evidence that the transformation is correct elsewhere. Therefore, inlier count, spatial distribution, residual structure, and coverage must be evaluated together.

Recommended acceptance logic:

- Require a minimum number of geometrically consistent inliers.

- Require adequate spatial coverage rather than only high inlier ratio.

- Reject transforms with implausible scale, rotation, determinant, or projective behavior.

- Inspect residuals for systematic spatial patterns.

- Mark individual tiles as failed instead of silently dropping them.

- Aggregate only control points that pass quality thresholds.

- Run a final global validation on independent or withheld control points when possible.

## 13. QGIS and GeoTIFF Output Requirements

QGIS is the practical visual verification layer. The registered output should be loadable beside the reference image without manual metadata repair.

| Property | Requirement |
| --- | --- |
| CRS | Preserve or explicitly transform into the selected target CRS |
| Transform | Correctly map raster pixel coordinates to geographic coordinates |
| Resolution | Record and intentionally select output pixel size |
| Bounds | Compute from the output grid, not blindly copy source bounds |
| Dimensions | Reflect the actual warped/resampled output grid |
| NoData | Preserve or explicitly define compatible NoData behavior |
| Compression | Use a configured lossless or appropriate compression strategy |
| Data type | Avoid accidental precision loss |
| Metadata | Record source files, method, model, and processing configuration where appropriate |

Visual checks should include: side-by-side inspection, transparency overlay, edge alignment, crater/terrain feature correspondence, and inspection of regions near tile boundaries.

## 14. Technical Hurdles and Solutions


| Technical hurdle | Root cause | Primary solution | Validation |
| --- | --- | --- | --- |
| System OOM | Whole-raster loading | Windowed reads + bounded concurrency | Peak RAM benchmark |
| Ray shared-memory | Large serialized arrays | File paths + windows | Object-store usage and |
| exhaustion |   |   | task payload size |
| Unbalanced keypoints | High-contrast clustering | Grid bucket filtering | Spatial heatmap of |
|   |   |   | retained points |
| False matches | Repeated textures / shadows | Ratio test + RANSAC | Inlier ratio and residual |
|   |   |   | analysis |
| Deep-model memory spikes FP32 + autograd + large tiles |   | Inference mode + precision control | Peak VRAM and tile-size |
|   |   |   | sweep |
| Bad global model | Terrain/parallax distortion | Model selection + local analysis | Residual map |
| Metadata loss | Pixel-only OpenCV workflow | Explicit geospatial writer | QGIS load/overlay |
| Worker filesystem failure | Path not visible on node | Shared/staged data design | Worker preflight |
| CPU oversubscription | Nested native threads | Ray resource limits + thread controls | CPU utilization benchmark |

## 15. Benchmarking and Experimental Plan

Do not evaluate the system only on whether it finishes. Measure accuracy, robustness, speed, and resource use.

## 15.1 Baseline experiments

| Experiment | Variables | Primary outputs |
| --- | --- | --- |
| Tile-size sweep | 512, 1024, 2048 | RAM, runtime, matches, RMSE |
| Overlap sweep | 0%, 10%, 20% | Boundary failures and runtime |
| Grid sweep | cell size and max features | Spatial coverage and inlier quality |
| Ratio threshold | e.g. 0.65–0.85 | Precision/recall trade-off |
| RANSAC threshold | multiple pixel tolerances | Inliers and residuals |
| Model comparison | Affine vs homography | Accuracy and stability |
| Classical vs deep | SIFT vs SuperPoint/LightGlue | Quality, runtime, memory |
| Concurrency sweep | 1, 2, 4 workers | Throughput and resource contention |

## 15.2 Dataset categories

- Easy: strong texture, modest viewpoint difference, stable illumination.

- Moderate: different acquisition time, moderate shadow variation, reduced texture.

- Hard: low contrast, repeated craters/terrain, strong illumination change.

- Geometrically difficult: significant viewpoint change or terrain relief.

- Stress test: very large GeoTIFFs on constrained RAM.

The key experiment is not 'does the deep model produce matches?' It is 'does the deep model improve the final registration quality enough to justify its compute and memory cost?'

## 16. Recommended Implementation Sequence

| Order | Build | Definition of done |
| --- | --- | --- |
| 1 | Raster loader | Large raster processed without full-array allocation |
| 2 | Tile metadata | Every tile maps unambiguously to source pixel/geospatial coordinates |
| 3 | SIFT + grid | Feature distribution is measurable and reproducible |


| Order | Build | Definition of done |
| --- | --- | --- |
| 4 | FLANN + ratio | Tentative matches generated with configurable threshold |
| 5 | RANSAC | Inliers, residuals and model diagnostics exported |
| 6 | Warp/export | Output opens correctly in QGIS with correct geospatial placement |
| 7 | Evaluation | Automated quality gate and summary JSON |
| 8 | Ray | Same results can be dispatched across multiple nodes |
| 9 | Deep matcher | Deep path improves hard-case benchmark results |
| 10 | Production CLI | One command reproduces a complete run |

This sequence prevents a common engineering mistake: optimizing infrastructure before proving algorithmic

correctness.

## 17. Risks, Blind Spots, and Technical Corrections

| Potential assumption | Why it is risky | Correct treatment |
| --- | --- | --- |
| gc.collect() solves memory pressure | Allocator/framework memory may remain | Use bounded memory architecture; measure |
|   | resident | actual peak usage |
| 10% overlap prevents all edge | Displacement may exceed overlap or features | Benchmark overlap and feature loss |
| problems | may still be absent |   |
| Affine transform update equals image | Metadata mapping and pixel resampling are | Define and execute the complete geospatial |
| registration | different operations | warp |
| Homography always handles terrain | Relief creates non-planar displacement | Validate residual spatial structure; consider |
|   |   | orthorectification/local models |
| FP16 always halves memory | Not all tensors/operators change precision | Measure actual peak VRAM/RAM |
| More matches means better | Repeated/incorrect features can increase | Prioritize geometrically consistent, spatially |
| registration | false correspondences | distributed matches |
| Ray automatically makes it faster | I/O, serialization and network overhead may | Benchmark single-node vs distributed execution |
|   | dominate |   |
| Deep learning is automatically | Models can fail or be slower on some imagery Compare against SIFT baseline on representative |   |
| superior |   | data |

The most important blind spot is geometric validity. Memory engineering and distributed execution are engineering problems; deciding whether the estimated transform is physically meaningful is the scientific problem. The latter must remain visible in the architecture.

## 18. Final End-to-End Workflow

Step 1 — Preflight: Validate files, CRS, raster dimensions, bands, NoData, model weights, hardware, and configuration.

- Step 2 — Tile planning: Generate deterministic windows with configured dimensions and overlap.

- Step 3 — Windowed reading: Read only the assigned reference and target windows.

- Step 4 — Preprocessing: Normalize or scale intensities as configured.

- Step 5 — Feature extraction: Run SIFT or the selected deep feature extractor.

- Step 6 — Spatial balancing: Apply grid/bucket filtering for classical features.

- Step 7 — Matching: Run FLANN + ratio test or the selected deep matcher.

- Step 8 — Geometric verification: Estimate the configured model with RANSAC and calculate inliers/residuals.

- Step 9 — Quality gate: Reject weak, clustered, implausible, or high-error tile results.


Step 10 — Aggregation: Combine accepted control points and estimate the appropriate global or local

transformation.

Step 11 — Warp: Resample the target raster into the intended registered grid.

Step 12 — Geospatial export: Write CRS, transform, bounds, resolution, dimensions, NoData and metadata correctly.

Step 13 — Evaluation: Calculate inlier statistics, RMSE, coverage, runtime and failure rates.

Step 14 — Visualization: Open the outputs in QGIS and inspect overlays, boundaries and residual problem areas.

Step 15 — Reproducibility: Store configuration, model version, software environment, metrics and output paths

with the run.

## FINAL ARCHITECTURE SUMMARY

| Phase | Core technology | Primary objective | Primary risk |
| --- | --- | --- | --- |
| Phase 1 | RasterIO + SIFT + FLANN + RANSAC | Correct memory-safe baseline | Bad geometry / weak |
|   |   |   | texture |
| Phase 2 | Ray | Distributed tile throughput | I/O, shared memory, |
|   |   |   | oversubscription |
| Phase 3 | SuperPoint + LightGlue / LoFTR | Improve difficult matches | VRAM/RAM and model |
|   |   |   | reliability |
| Phase 4 | CLI + YAML + JSON + QGIS | Reproducible production workflow | Metadata/validation failures |

## FINAL DESIGN PRINCIPLE

The finished system should be judged on four independent dimensions: correctness (the geometry is valid), robustness (hard scenes do not silently fail), efficiency (memory and compute stay within hardware limits), and reproducibility (the same configuration produces auditable results).

A successful implementation is therefore not merely a script that aligns two images. It is a measurable registration system in which every major stage has bounded resource usage, explicit inputs and outputs, diagnostics, quality gates, and a clear relationship to geospatial coordinates.
