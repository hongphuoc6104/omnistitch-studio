# Unpaired 4X feasibility pilot

`prepare.py` creates lossless, nonoverlapping 256-pixel crops from the central
80% of one native sensor FOV per patient/domain. It validates patient splits,
source MD5, path containment and cross-identity patch hashes. Native Laplacian
variance and foreground heuristics are recorded, not presented as tissue labels.
The small pilot intentionally avoids mosaic seams; main mosaic-based experiments
must retain owner/3×3 transform provenance and use a separate dataset version.

```sh
python tools/virtual_stain_pilot/prepare.py --orders private_orders.json \
  --source-root /private/source --output /private/new_dataset
python tools/virtual_stain_pilot/train_adain.py --dataset /private/new_dataset \
  --output /private/pilot --steps 1000 --seconds 2700
```

Training requires a CUDA-enabled PyTorch environment, refuses accidental CPU
execution, and is bounded to 1000 updates / 45 training minutes. Only the decoder
is fine-tuned. Sampling balances patients separately in THO and H&E; validation
references are fixed TRAIN H&E crops. Test data is rejected. PNGs are previews
clamped to [0,1]; `.npy` predictions retain the numerical output. Checkpoints
include decoder, optimizer, RNG state and configuration; they contain data-derived
weights and must remain private. This script saves recoverable states but does
not yet expose an automatic resume command.

AdaIN is a **style-transfer baseline**, not validated virtual histology. A good
style match does not establish correct staining of nuclei or cytoplasm. Compare
held-out native structures and the pretrained step-0 baseline before expanding.
The identity term is a project experiment, not an exact paper reproduction.

References:

- Huang & Belongie, [AdaIN, ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.html).
- [Author implementation](https://github.com/xunhuang1995/AdaIN-style).
- The runner uses the MIT [unofficial PyTorch port](https://github.com/naoto0804/pytorch-AdaIN)
  at revision `47950d0e6656a95a80a4b105c4c0f58d38ef785c`, public weights
  from release `v0.0.0`, and retains its license and asset hashes in run provenance.

No private patient IDs, source paths, manifests, images or weights belong in Git.
The existing unit suite includes a synthetic file→native-crop→PNG→manifest E2E
test, pixel equality, checksum rejection, split leakage, native focus and path
containment. CUDA training must additionally be exercised on a disposable pilot
runtime; passing CPU preparation tests alone does not validate the GPU trainer.
