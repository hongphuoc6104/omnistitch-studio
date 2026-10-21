# P3 full-resolution exposure-only derivatives

`estimate_p3_gains(manifest, proxies)` reproduces the approved pilot's P3 fit:
384×216 raw RGB proxies, luminance Gaussian smoothing for estimation only,
maximum 160 accepted overlap edges, alternate fit/held positions, log-gain least
squares with identity regularization, median anchor, gain bounds [0.75,4/3].
Unsupported sources keep identity; disconnected photometric evidence is reported.
This is relative camera code-space correction, not physical calibration, stain
normalization, an independent patient-level evaluation, or a focus remedy.

`export_p3_mosaic` multiplies each original baseline RGB pixel by its original
owner's gain. Pixel positions, dimensions and ownership remain unchanged. Unlike
the native float pilot replay, the baseline mosaic has already been quantized to
uint8; applying gain to it differs only by that inherited rounding. The pilot
recorded native P0 replay MAE below 0.3 camera code values. No additional warp,
focus selection, blend, sharpening, field correction or deblurring is introduced.

The derivative is a **Deflate-compressed, tiled float32 RGB BigTIFF**. Compression
is lossless. Values above 255 are retained instead of clipping one color channel;
values are not automatically 0..1. Downstream readers must honor these units.
`read_exposure_region` decodes only intersecting tiles and preserves float32.
Many ordinary image viewers do not interpret this format's display range; the
JPEG preview clamps to 0..255 for display and its over-range fraction is recorded.
The exposure manifest references immutable baseline/owner hashes, transforms and
source gains. Original masks/owner maps remain at the baseline location.

Tile working sets, decode buffers, preview dimensions and worst-case staging disk
are bounded before allocation. Publication is staged; failures leave no partial
final TIFF. Input/output containment is enforced through the workspace resolver;
output cannot share the baseline directory. A successful export is never marked
accepted for training. Synthetic tests cover exact owner-wise gain application,
over-range RGB retention, unchanged geometry and input hashes, edge crop reads,
missing evidence, known exposure recovery, memory guards and failed publication.

The application UI and legacy blend paths are unchanged. The full 10X task uses
separate output directories and component-local manifests, including disconnected
components already present in the baseline. Original data are never overwritten.
