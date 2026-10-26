# Paired 4X research readiness

These tools implement two bounded pilots, not validated virtual histology:

* **B:** the official SwinIR RGB backbone at scale 1, with a small
  60-channel, three-group configuration, trained from scratch with L1.
* **C:** a **project conditional DiT variant**, trained from scratch with an
  epsilon diffusion objective, a frozen public VAE, and concatenated
  noisy-target/source latents. The DiT is deliberately small (192 hidden
  channels, six blocks). This is not the pretrained D-VST, its pathology/tone
  encoders, its curriculum or a reproduction of its published results.

Both use transformers; they differ in direct deterministic RGB regression vs
iterative stochastic latent generation. No GAN, super-resolution, sharpening or
stain normalization is introduced by these tools. Actual backbone/licensing
assets are downloaded at explicitly pinned revisions, SHA256 checked **before
import**, and kept outside Git. SwinIR is Apache-2.0; upstream DiT code is
CC-BY-NC-4.0. Preserve their downloaded licenses with experiment evidence.

## Data is the gate

An anatomical reviewer must establish correspondence at the loss scale. Adjacent
physical sections have no cell-level ground truth simply because they share a
patient or have a good outer contour. Capture intersection is only observed
coverage. A candidate pack is inspectable, but cannot enter these fine-scale
pilots. Coarse correspondence cannot be silently promoted to fine supervision.

`data.py` accepts lossless bounded PNGs (up to 512 pixels per side), full 3×3
native transforms, original native boxes, SHA256, patient split and separate
capture/supervision masks. The immutable `approved_release` must contain at
least two train and one validation patients; test is rejected. Reviews explicitly
name fit and held-out structures, approved B/C objectives, mask inspection,
anatomical identity and held-out spatial coverage. A 4px maximum held-out error
is an **engineering ceiling for this pilot**, not a diagnostic criterion or a
sufficient anatomical validation by itself. The current common B/C comparison
requires a completely approved patch: a VAE mixes surrounding context, so simply
masking a latent cell does not remove all information from excluded tissue.

Do not set review flags from Dice/MI/capture overlap. Do not manufacture a
nonzero supervision mask for an unresolved pair. A pending candidate pack uses
all-zero masks. To add partially reviewed regions later, crop a new native patch
version within the approved support, preserve geometry/source provenance, and
review that version. Do not resize or repeatedly warp saved H&E targets.

```sh
# Inspect/checksum candidates separately; final release command rejects them.
python -m tools.paired_stain.data --dataset /private/candidate_dataset \
  --reviews /private/independently_reviewed_masks.json --output /private/new_release
python -m tools.paired_stain.data --dataset /private/new_release
```

`publish` copies unchanged source PNGs, supplied reviewed masks and QC into a
new directory, validates it and writes its manifest last. Existing outputs are
never replaced. Incomplete outputs after interruption require inspection.

## Prepare before allocating GPU

Use an isolated Python 3.12 environment with CUDA torch/torchvision (reference
CPU E2E: 2.5.1/0.20.1), and `pip install -r
tools/paired_stain/requirements.txt`. Do not replace Colab CUDA Torch with a CPU
wheel. Pin/record actual installed versions in experiment evidence.

`python -m tools.paired_stain.bootstrap_colab --root /content/research_env
--accelerator cuda` creates a venv **without inheriting notebook packages**,
using Python 3.12, matching Torch 2.5.1/torchvision 0.20.1 CUDA 12.1 wheels and
the pinned research requirements. Use `--accelerator cpu` for CPU diagnostics.
It leaves notebook Torch unchanged. This avoids optional dependency conflicts
seen on stock Colab (a plain VAE import pulled an incompatible torchao API).
The environment setup downloads its own wheels once; reuse it within that
disposable runtime rather than reinstalling for every phase.

```sh
python -m tools.paired_stain.prepare assets --branch B --assets /private/assets
python -m tools.paired_stain.prepare assets --branch C --assets /private/assets \
  --vae-cache /private/vae_cache
# This works with pending candidates and requires no model training.
python -m tools.paired_stain.prepare codec --dataset /private/candidates \
  --output /private/codec_diagnostic --vae-cache /private/vae_cache \
  --device cuda --limit 16 --seconds 600
```

Codec reconstruction is a **same-image** diagnostic on THÔ and H&E, separately
stratified by train/validation. Original/reconstructed panels, unclipped float
arrays and same-image MAE are retained; there is no automatic approval. C needs
a separate `approval` JSON with reviewer/evidence and the report SHA. Its report
must cover **all unique input images of the eventual release**, with unchanged
reconstruction files and the pinned VAE revision. A limited eight-image codec
sample does not approve the remaining cohort. These measurements are not sensor
focus metrics; use original sensor crops for Laplacian focus inspection.

`benchmark.py` can exercise the full B/C forward/backward on candidate RGB before
labels are ready. It needs CUDA, performs **zero optimizer updates**, saves no
weights and records hardware, peak VRAM and timings after three warmup passes.
It cannot establish staining quality or a codec/correspondence approval.

## Two independent pilots

```sh
python -m tools.paired_stain.train --branch B --dataset /private/new_release \
  --assets /private/assets --output /private/B_run --steps 1000 --seconds 2700
python -m tools.paired_stain.train --branch C --dataset /private/new_release \
  --assets /private/assets --vae-cache /private/vae_cache \
  --codec-approval /private/codec_approval.json \
  --output /private/C_run --steps 1000 --seconds 2700
```

All data gates run before GPU/model allocation. Real-data CPU training is
refused. Each run uses microbatch 1, patient-balanced sampling, accumulation 4,
AdamW 1e-4, FP16 with scaling, finite-gradient checks, no test images and a
45-minute/1000-update ceiling. Up to three minutes (20% for shorter runs) is
reserved for held-out evaluation within the same total budget. An in-progress
update that passes the training deadline is discarded before the optimizer
step. Inspect actual validation counts: unexpected setup/I/O delays can still
reduce coverage within the hard total deadline. Individual CUDA kernels
cannot be preempted by a Python deadline. Use a runtime/controller watchdog and
allow a short checkpoint/copy grace period.

Models are not expected to converge in the first pilot. Measure 20–50 warmup
updates on the assigned GPU before budgeting main training; two Pro accounts do
not pool VRAM or guarantee identical hardware. CUDA feasibility is still a
separate empirical check after CPU E2E has passed.

The same release hash and deterministic validation IDs serve both branches;
C uses three fixed seeds, never a selected best seed. Preview PNGs are clipped
for display only and unclipped `.npy` predictions are retained. Paired L1 in the
output is an alignment-dependent diagnostic, not cell-level accuracy. Additional
pathologist/structural review is needed before ranking models as virtual stains.
Serial-section texture differences remain even in a reviewed patch.

Checkpoints include model, optimizer, scaler, completed step, Python/Torch/CUDA
RNG, code hashes, configuration and dataset hash. Resume a trusted private
checkpoint with identical learning configuration and a **new output directory**:
`... --resume /private/B_run/checkpoint.pt --output /private/B_resumed --steps
1000`. Identity mismatches are rejected. Atomic periodic checkpoints recover up
to the last saved update after interruption, not unsaved updates in flight.
Keep data-derived weights private and verify Drive backups before stopping only
task-owned runtimes. Drive consent remains interactive in the user's terminal.

## Verification

Default unit tests cover file→candidate rejection→reviewed release, exact PNG
preservation, native coordinate/header checks, patient/pixel leakage, checksum
and path containment, supervision/capture semantics and fit/holdout separation.
An additional isolated CPU CI job downloads the pinned public backbone code,
then runs both tiny models end-to-end on explicitly **synthetic** fixtures. It
verifies finite forward/backward, conditioning gradients, evaluation and
bit-identical model parameters for interrupted-vs-continuous checkpoint resume.
The synthetic C fixture codec is never available for real-data training.

```sh
CTUP_TEST_TORCH=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s tests -p test_paired_stain.py -v
```

Sources: [official SwinIR](https://github.com/JingyunLiang/SwinIR),
[official DiT](https://github.com/facebookresearch/DiT),
[D-VST](https://github.com/yangshurong/D-VST), and
[public VAE model card](https://huggingface.co/stabilityai/sd-vae-ft-mse).
The VAE was trained on natural images, so its preservation of microscopy
structures must be checked rather than assumed.
