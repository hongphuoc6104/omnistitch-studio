# Serial-section pair audit (THO fixed, H&E moving)

This opt-in research path keeps registration, capture coverage, anatomical
correspondence and training acceptance separate. Different physical sections
can have missing, torn or genuinely different anatomy. Good silhouette Dice,
mutual information or feature residuals do not establish cell-level ground truth.

`backend.serial_pairing` provides bounded original-mosaic proxy sampling, exact
3×3 proxy/native coordinate conversion, held-out landmark diagnostics and native
pair ROI/full derivative export. `tools.serial_pairing.registration` proposes
SIFT/RANSAC and silhouette initializations, with optional affine mutual-information
refinement. No method automatically sets `trainingAccepted` to true.

## Coordinate and source contract

THO is the reference canvas. Stored `HE_native_to_THO_native` maps H&E source
pixel centers **forward** to THO pixel centers. The inverse is stored explicitly.
Always use `warpPerspective`, retaining the complete 3×3 projective contract.

For a proxy produced by the fixed-grid sampler, native coordinates are
`(proxy + 0.5) * native_size / proxy_size - 0.5`. Thus the native transform is
`P_fixed @ H_proxy @ inverse(P_moving)`. Tile-pasted legacy JPEG previews are
screening-only: regenerate original-mosaic proxies before claiming exact native
coordinates. Proxy smoothing/CLAHE is restricted to registration representations;
it is never a clarity measurement or a modification of the training RGB source.

The native exporter reads cached tiled uncompressed uint8 original dataset TIFFs.
It samples a bounded source enclosure for each fixed tile, applies one geometric
interpolation, writes disk-backed maps and streams tiled BigTIFF. Memory and
padded-disk guards run before output buffers. Raw input files are hash-checked
and not overwritten. A nonempty destination requires a new version rather than
silently replacing an existing pair. File publication is staged and rolled back
on failure. Runtime/Drive persistence must additionally verify every output hash
before a run-specific completion marker is written.

## Artifacts and interpretation

- Immutable THO/H&E references: owner account, logical Drive path, hashes, patient,
  split, group/component and original provenance manifest.
- `HE_aligned_to_THO.tif`: full global-alignment review derivative. Original THO
  remains the fixed image; one unchanged source can be referenced by several
  candidate pairs without duplicating it.
- `capture_overlap.tif`: intersection of camera observation coverage. White glass
  is not padding, and this mask does **not** assert biological correspondence.
- `pair_manifest.json`: forward/inverse 3×3, dimensions, mask hashes, source and
  derivative hashes, interpolation, pending review and `loss_mask_available=false`.
- The run-level worker adds proxy comparisons, overlays and native ROI pairs with
  original fixed coordinates. Native ROI checks happen before any paired training.

Anatomical ignore/loss masks and independent annotations belong to a subsequent,
versioned review. Do not infer them from capture overlap. Missing annotations
remain pending; optimizers' matches/metrics must never be relabeled independent
anatomical landmarks. Reflection proposals need explicit review. No same-name
THO_A/H&E_A identity or one-to-one assignment is assumed.

## Optional registration dependency and evidence

The core native coordinate/export module needs only existing repository
dependencies. Affine MI refinement additionally uses `SimpleITK==2.4.0` in the
experiment's isolated CPU environment; it is not a mandatory Studio dependency.
SimpleITK optimization maps fixed→moving for resampling, so its transform is
inverted for the stored moving→fixed convention. Mattes MI is an optimization
diagnostic, not a success certificate. Nonrigid fields are **not implemented** in
this global stage and must never be substituted into the 3×3 affine/projective
contract. A separate residual review sampler is described below; it does not
turn global proposals into accepted pairs.

The core tests cover file→native ROI alignment, full projective third-row use,
pixel-center lifting, immutable sources, streaming publication, repeat refusal,
bounded proxy reads, capture-mask meaning, landmark separation and guards.
Real-data pilot evidence is private; synthetic tests do not validate anatomy.

Primary references:

- [SimpleITK registration overview](https://simpleitk.readthedocs.io/en/v2.3.0/registrationOverview.html)
  for transform directions, metrics and multi-resolution registration.
- [SimpleITK ImageRegistrationMethod](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ImageRegistrationMethod.html)
  for metric masks, Mattes mutual information and optimizer settings.
- [VALIS, Nature Communications 2023](https://www.nature.com/articles/s41467-023-40218-9)
  for pathology-series registration context and serial-section limitations.

Feature, mask, optimizer and QC thresholds in this implementation are experimental
engineering choices. They are not claimed to be medical acceptance standards or
an exact reproduction of VALIS.

Affine MI uses a one-proxy-unit maximum physical step with per-iteration learning-rate estimation. This avoids the reproduced unrestricted-line-search support loss. Registration silhouettes fill enclosed lumen holes to avoid modality-dependent dark-area scale bias; they remain unsuitable as anatomical loss masks. A private-file optional MI regression is retained in `tests/test_serial_pairing_optional_mi.py`; its real fixture is never published.

Continuous outline initialization uses multi-resolution ECC on clipped signed
silhouette distances before multimodal affine MI. It retains each stage's 3×3
matrix separately. ECC is affine-only here; projective inputs are rejected rather
than truncated. Its inverse-sampling direction follows the [OpenCV ECC API](https://docs.opencv.org/doc/doxygen/html/dc/d6b/group__video__track.html).
The synthetic direction test recovers a known asymmetric affine and rejects
homographies. Outline improvement is still not internal anatomical validation.

`assignment_hypotheses` enumerates bounded one-to-one alternatives and unmatched
sections. For two THO and two H&E inputs it retains both AA+BB and AB+BA, rather
than matching letter suffixes or greedily choosing each source's best edge. The
whole graph, alternative assignments and score gap are diagnostics only. Physical
fragment mapping may invalidate a one-to-one assumption. Tests cover a greedy
trap, crossed suffixes, missing sections and a hard enumeration budget. No branch
claims anatomical identity or automatically accepts a pair for training.

## Separate backward residual review

`backend.serial_deformation` inspects bounded native ROIs with an explicitly
separate float32 displacement field. Its direction is **THO proxy → globally
aligned H&E proxy**, with additive displacement in fixed-proxy pixel units.
For fixed native point `p`, fixed native/proxy scale `s`, and global forward
matrix `H`, first compute `q = (p + 0.5) / s - 0.5`, then
`r = (q + u(q) + 0.5) * s - 0.5`. Apply the complete `inverse(H)` to homogeneous
native point `[r.x, r.y, 1]` to obtain the original H&E sampling location.
Thus convert `p` to proxy centers, add the backward residual, convert back to
native centers, then apply `inverse(H)`. The implementation keeps the
full 3×3 matrix and currently rejects projective globals in this residual path.
The global-only path continues to support projective matrices.

The field must be contiguous float32, finite, at most 1600 pixels per side,
within a configured displacement cap and have positive **sampled**
`det(I + gradient(u))`. These checks do not prove continuous invertibility or
biological correctness. Half-pixel edge extensions replicate field boundaries.
Nonlinear source support is bounded using all mapped ROI samples, including
interior extrema. Pre-allocation budgets cover coordinates, field and source
support. Original H&E RGB is sampled **once** after composing the two mappings;
double-warped proxy RGB is suitable only for preview. No sharpening, stain-color
normalization or clarity measurement is performed here.

The optional private B-spline pilot uses
[SimpleITK's multiresolution registration framework](https://simpleitk.readthedocs.io/en/latest/link_ImageRegistrationMethodBSpline3_docs.html).
Its mesh, optimizer bounds, metric masks and same-image reserved-position
diagnostics are experimental choices, not published acceptance thresholds.
Fields, transforms, input hashes and native comparisons remain versioned review
artifacts. Independent anatomical landmarks and anatomical ignore/loss masks
are still required before paired training; positive Jacobians or improved MI
never supply those annotations.

Tests exercise file→field→native ROI sampling, backward sign and half-pixel
scale, affine composition, immutable source hashes, interior support extrema,
capture masks, invalid fields, folds, displacement limits and memory guards.
