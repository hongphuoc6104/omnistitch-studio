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
