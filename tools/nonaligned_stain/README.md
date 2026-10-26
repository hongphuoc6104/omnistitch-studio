# Nonaligned 4X experiments

These two **project variants** let research proceed without declaring serial
sections to be exact THÔ–H&E pairs. Fine-paired runners and their data guards
remain separate and unchanged. Nonaligned outputs are hypotheses for review,
not verified nuclear/cytoplasmic staining or diagnostic images.

## B2: deterministic contextual SwinIR

Official scale-1 SwinIR is initialized as the observed input by zeroing its final
residual convolution. Training uses a frozen official ImageNet VGG19 with the
required RGB mean/std normalization, features at relu3_1 and relu4_1, and:

`CX + 10 × source self-similarity + 5 × RGB palette moments + 5 × same-HE identity + 0.1 × range penalty`.

CX compares feature sets without a pixel alignment assumption. Relative cosine
distances, contextual weights and target feature coverage follow the principle
of [Mechrez et al., ECCV2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Roey_Mechrez_The_Contextual_Loss_ECCV_2018_paper.html)
([author code](https://github.com/roimehrez/contextualLoss)). Pairwise source
self-similarity follows the principle of
[STROTSS](https://arxiv.org/abs/1904.12785), with token-count rescaling to keep
its numeric scale useful in this experiment. This is not a reproduction of
either author's full training system. The combination, coefficients, frozen
VGG19 and SwinIR are project choices to test, not validated histology choices.

The HE reference during training is sampled from the **same train patient**,
without matching section suffix, field or pixels. This is weak patient-level
association, not a paired target. The identity term compares an HE image with
itself. No THÔ–H&E pixel L1/PSNR is used. Inference needs only THÔ; displayed
H&E references are fixed TRAIN images, never a corresponding ground truth.

Feature sets are bounded to a 16×16 pooled grid (256 tokens), avoiding quadratic
full-resolution allocation. B2 runs the complete model/feature/backward path in
FP32: a real T4 run had nonfinite gradients at its first FP16 backward despite
FP32 pairwise distances. The finite-gradient guard rejected that update. C2 uses
FP16 AMP for its independently optimized epsilon losses. Precision is explicit
in the run identity; failed FP16 outputs are not counted as completed training.
Pooling and ImageNet features may miss small histological changes; inspect
native outputs rather than using the objective as proof of fidelity.

## C2: two domain DiTs and DDIB-inspired transport

[DDIB](https://suxuann.github.io/ddib/)
([author implementation](https://github.com/suxuann/ddib)) trains source and
target diffusion models independently, using source-to-noise then noise-to-target
transport. This prototype uses two **independent small unconditional DiTs**,
a shared frozen public VAE, and deterministic finite-alpha DDIM integration.
It uses none of the author's original image generator backbones, guidance or
pretrained image models. It is a latent DiT adaptation, not an author DDIB
reproduction or the earlier conditional D-VST-inspired branch.

Only TRAIN images enter the latent cache. Each domain has its own sample,
epsilon-noise target, optimizer, scaler and EMA. One reported step is one
optimizer update **per domain**, i.e. two updates total. EMA warms with
`min(0.999, (1+step)/(10+step))`. No paired similarity objective is needed.

Translation is deterministic: invert THÔ with its score field, then decode the
same finite noisy latent with the HE field. Fifty steps are used in each
direction, without clamping. The terminal alpha remains positive. Numerical
integration introduces cycle error; small cycle error does not certify tissue
identity. Separate controls retain VAE-only reconstruction, same-domain
invert/decode and cross-domain cycle images, isolating codec and solver changes
from a change of stain domain. Do not interpret these as reconstruction of the
true chemistry of an unobserved HE stain.

## Data and operation

`data.py` freezes unchanged original-sensor PNG crops from an already audited
dataset. The texture-rich core requires `structured_fraction >= 0.75` using
existing native QC; this is a technical selection, not a calibrated tissue
mask. Patient/source/pixel leakage, file hashes, sensor coordinates and both
train/validation domains are checked. Test patients are rejected. A release
explicitly states `paired=false` and `ready_for_unpaired_pilot=true`, while
`trainingAccepted=false` continues to mean no clinical/fine-pair acceptance.
No zero/unknown correspondence masks are turned into positive masks.
C2 can keep patients with only one observed domain. Same-patient TRAIN style
availability is checked specifically for B2; it must not become an accidental
pairing requirement for independently trained C2 marginals.

```sh
python -m tools.nonaligned_stain.data --source /private/audited_sensor_dataset \
  --output /private/new_unpaired_release
python -m tools.nonaligned_stain.train --branch B2 --dataset /private/new_unpaired_release \
  --assets /private/assets --output /private/B2_run --steps 400 --seconds 900
python -m tools.nonaligned_stain.train --branch C2 --dataset /private/new_unpaired_release \
  --assets /private/assets --vae-cache /private/vae_cache \
  --output /private/C2_run --steps 1000 --seconds 900
```

Use the isolated research environment and pinned public backbone assets from
`tools.paired_stain`. VGG19 is obtained through torchvision's official weight
enum/hash-checked URL; its full downloaded SHA is recorded in run.json. Public
weights need not be copied into private result archives. Keep code/license and
weight identities, all private checkpoints, native inputs, metrics and float
predictions with experiment evidence.

Real-data training requires CUDA. CPU runs are restricted to explicitly
synthetic fixtures with tiny backbones and an explicit fixture feature/codec
path, never presented as VGG/VAE validation. Both methods use the same patient
split and deterministic held-out THÔ crop selection as the existing style
baseline. Validation references are train-only. C2 transport is deterministic,
so there is no best-of-seeds selection. Preserve unclipped numerical outputs;
PNG clipping and grayscale correlations are diagnostics, not histology scores.
There are too few independent held-out patients to rank methods by patch FID.

Runs are bounded to 1000 steps / 45 total minutes. Up to three minutes is
reserved for held-out evaluation. Checkpoints include live/EMA weights,
optimizer/scaler, completed steps, RNG and data/code identity. Resume trusted
private checkpoints with `--resume` into a new output directory. A controller
must copy verified checkpoints/results to Drive and stop only owned runtimes.
No long training run should be inferred from a successful synthetic test.

## Round 2: bounded diagnostics and B3 ablations

The existing source dataset remains unpaired. New experiment options are
`--seed`, a complete JSON `--loss-weights` mapping, `--gradient-diagnostics`,
`--eval-all` and `--compact-evaluation`. Defaults retain the original loss
coefficients. Seed, coefficients and diagnostic/evaluation policy are recorded
in checkpoint identity; changing the learning configuration is not resume.
B2 remains fully FP32. Probes at steps 0/20/100/400 measure weighted gradients
at the prediction, residual head and first shared convolution without optimizer
updates. The separate HE identity forward has no gradient with respect to the
THÔ prediction, but does have gradients through shared model parameters.
A zero-initialized residual head blocks upstream gradients initially; the
probe reports that fact instead of treating it as a failed training step.

Initial/intermediate image checks use the fixed quick set. `--eval-all` writes
final predictions for all held-out THÔ crops, preserving the quick-set TRAIN
style references. Compact mode retains float predictions and viewing panels;
source images remain recoverable by immutable dataset file/hash identities.

`python -m tools.nonaligned_stain.diagnose --help` evaluates a trusted private
checkpoint without any optimizer. An explicit checkpoint SHA, matching native
release and pinned backbone/codec revisions are required. It retains all THÔ
outputs and patient-level summaries. C2 additionally evaluates codec round trips
on both validation domains, fixed-noise epsilon prediction on separate TRAIN
and validation records, same-domain/cross-domain/cycle controls at 25/50/100
steps, and 16 predefined Gaussian seeds per domain. Generated marginal images
have no corresponding observed target. Positive terminal alpha is recorded:
noise moments and Gaussian-only sampling do not prove density correctness.

The one-terminal `terminal_round` helper allocates a uniquely owned ordinary
T4 only when invoked. The user consents directly through terminal drivemount;
consent is bounded to five minutes. Input archives are read-only from the
already verified Drive backup, hashed before selective streaming extraction.
An isolated environment runs C2/B2 checkpoint diagnostics, then four B3
screenings of at most 1,000 updates each. B3-T changes either CX or palette,
chosen from the control's TRAIN step-20 head gradients before seeing validation
results. This static perturbation is a project experiment, not GradNorm or a
validated histology prescription. Native review is required before choosing a
seed confirmation or C3 intervention; this helper starts neither automatically.

The worker and controller reserve transfer time inside a 150-minute total T4
ceiling. Results retain code, native predictions, checkpoint/optimizer/RNG,
metrics, pinned backbone source/licenses and environment identities. SHA-checked
Drive publication and DriveFS flush precede success; output parts are verified
locally, logs are collected on success/failure and only the owned runtime is
stopped. Reinvocation reuses verified evidence; a failed cleanup retries the
same owned session without retraining. No main training or new account choice
is authorized by this helper. The project user will supply two main-training
accounts when a suitable final protocol is ready.

Tests for this workflow include real fixture PNG→train→checkpoint→diagnostic
loops, nonmutating gradient probes, all-validation coverage, a Gaussian-score
solver control, deterministic bundle creation, archive→parts integrity, and a
fake CLI boundary for consent failure/cleanup/idempotence. Fake CLI tests do
not establish real OAuth, GPU, DriveFS or histology performance.

If Drive publication fails after computation, the complete local tar/part
manifest is retained. The controller still verifies/downloads the evidence and
stops its GPU, then reports the missing durable backup. Repair the backup with
a CPU-only task; an existing verified result must not trigger retraining.

### Explicit transport fallback after a DriveFS outage

`input_transport=local_parts` uses unchanged native crops and complete trusted
checkpoints already verified locally. `inputs.pack` streams a deterministic tar
and 8 MiB parts; every input part is checked before GPU allocation and every
member/checkpoint/native manifest is checked on the runtime. No Drive consent
is requested in this explicitly configured mode. Geometry, pixels, learning
objectives and patient splits are unchanged. Unrecognized modes, corruption,
links, traversal and missing members are rejected.

With `allow_local_verified_completion=true`, the controller may finish only
after full result SHA/part verification on the local data partition. It records
`Drive_backup_verified=false`, `Drive_backup_pending=true` and prints
`ROUND2_LOCAL_COMPLETE_DRIVE_PENDING`; this is not durable Drive publication.
The GPU is stopped and any later Drive repair/backup uses a CPU task. Repeated
invocation reuses the same verified local results instead of retraining. The
normal Drive-backed mode still requires verified Drive publication.
