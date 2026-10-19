# Bounded dataset mosaics

This opt-in Python API prepares microscopy research datasets from overlapping
RGB camera frames. It does not replace the Studio's existing interactive/auto
blender or imply that a successful export is ready for training.

## Registration and export

```python
from backend.dataset_registration import register_camera_frames
from backend.dataset_mosaic import export_dataset_mosaic

registered = register_camera_frames(
    [{"id": "frame-a", "path": "inputs/a.tif"},
     {"id": "frame-b", "path": "inputs/b.tif"}],
    workspace_root="/project",
)
metadata = export_dataset_mosaic(
    registered["sources"], "/project/output/section-a", "/project",
    edges=registered["edges"], memory_budget_bytes=512 * 1024 * 1024,
)
```

Use one patient/physical section or fragment/stain/objective/acquisition mode per
job. The caller owns this grouping and the patient partition. All source/output
paths must resolve inside the workspace. Supported inputs are bounded,
single-plane RGB uint8 camera frames; other depths/channels are rejected rather
than silently normalized. Supply correct scale/calibration separately: no MPP is
inferred from a filename, TIFF DPI, or objective label.

Sources may instead be supplied with reviewed `sourceToWorld` transforms: nine
row-major entries, source pixel to world pixel, and the real match graph edges.
Disconnected graphs, singular transforms and projective horizons are rejected.
The exporter uses full 3x3 perspective warps and records output↔world transforms.

Registration reads one bounded frame at a time. A proxy and at most 2,000 SIFT
features per frame are retained by default. Correspondences are converted back
to native source pixels before RANSAC similarity estimation. All-pairs matching
is used without reliable stage adjacency metadata; this remains quadratic.
Spanning-tree transforms and loop residuals are diagnostics, not bundle
adjustment or an independent proof of geometry. Persist the returned registration
diagnostics alongside the mosaic manifest.

## Selecting captured detail

The exporter computes variance of a grayscale Laplacian on native crops
(default 256 px, sampled every 128 px). It never measures focus on a preview or
on warped/downsampled RGB. A fixed soft edge prior scales scores by 0.8–1.0
within the outer 32 source pixels. The dominant term is native focus, so a clear
corner can beat a blurry center. Source IDs break ties deterministically.

This is a baseline ranking cue sensitive to noise, exposure and tissue content;
it is not a biological focus ground truth. Inspect artifacts, texture-poor regions
and same-world-position comparisons before choosing a production policy. The
RGB output selects one captured source per pixel, with geometric interpolation.
There is no defringe, color balancing, sharpening, deblurring or flat-field
correction in this path. Hard ownership transitions can remain visible. Compare
photometric corrections with ownership fixed in a separate experiment.

## Artifacts and limits

- `mosaic.tif`: tiled RGB uint8 BigTIFF.
- `owner.tif`: uint32 owner IDs; 0 means unobserved. Manifest IDs map back to source
  paths, hashes, dimensions and 3x3 transforms.
- `valid.tif`: observed coverage, **not a tissue/usable-quality mask**.
- `seam.tif`: one-pixel source-ownership boundaries for downstream QC/exclusion.
- `quality_margin.tif`: relative best/second-best score gap, not calibrated
  confidence. A sole observation is 255 even if blurry; unobserved is 0.
- `manifest.json`: algorithm parameters, provenance, geometry QC, output hashes
  and measured maximum tile/ROI working estimates.

Only bounded tile/ROI arrays and small focus grids are allocated. Uncompressed
per-frame caches permit region reads of originally compressed camera TIFFs.
Output arrays use disk-backed maps; dirty pages are flushed/reclaimed per row on
OSes supporting `madvise`, including Colab Linux, during rendering and TIFF
writing. The byte budget is a conservative working-array estimate, not a hard
process RSS limit covering codecs, Python, or OS caches. Benchmark actual peak
RSS on the target runtime. Disk preflight includes caches and staged output.
Publication uses the existing artifact-set rollback mechanism.

Independent landmarks may be supplied as dictionaries containing `source_a`,
`xy_a`, `source_b`, `xy_b`. Median/p95 distances are in world pixels; a 1/3 pixel
target is reported without implying cell-level correctness or sufficient sample
size. `trainingAccepted` remains false: successful registration/coverage does not
replace independent geometry and histology QC. Serial THO/H&E sections remain
weak pairs; FOV stitching does not turn them into same-cell ground truth.

## Regression evidence

A deterministic file→pipeline→TIFF counterexample positions the sharp source's
corner against a blurred source's center. The legacy auto output differs from
the sharp captured crop by mean absolute RGB error 20.472; native-focus export
reduces that error to 0.0 at the tested location, without sharpening. This checks
selection behavior on known synthetic detail, not accuracy of real nuclei.
Tests also exercise chunk/order invariance, full projective geometry, uint32 owner
IDs, missing coverage, rollback, resident-page reclamation and decode/ROI/disk
guards. Real-data selection and registration still require separate validation.
