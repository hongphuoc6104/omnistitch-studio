"""Bounded AdaIN decoder fine-tuning: unpaired technical baseline, not H&E GT.

Uses the explicitly pinned, MIT-licensed naoto0804 PyTorch port of Huang and
Belongie (ICCV 2017). That port is unofficial; its architecture/weights are kept
separate from project code. Source and weight hashes are recorded for each run.
Torch is optional and is imported only when this GPU-only experiment is invoked.
"""

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

UPSTREAM = "47950d0e6656a95a80a4b105c4c0f58d38ef785c"


def download(url, path):
    """Fetch a pinned public artifact; retain its SHA256 in run provenance."""
    if not path.is_file():
        temporary = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as target:
            while block := source.read(4 * 1024 * 1024):
                target.write(block)
        temporary.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(args):
    """Train with patient-balanced unpaired sampling and a sealed test split."""
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise RuntimeError("GPU pilot refuses an accidental CPU training run")
    if not 1 <= args.steps <= 1000 or not 1 <= args.seconds <= 2700:
        raise ValueError("Pilot budget exceeds 1000 steps / 45 minutes")
    torch.set_num_threads(2)
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    root, output = Path(args.dataset).resolve(), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["records"]
    owners, pixel_owners, buckets = {}, {}, {}
    for row in rows:
        patient, split, domain = row["patient_id"], row["split"], row["domain"]
        if split not in ("train", "validation") or domain not in ("THO", "HE"):
            raise ValueError("Invalid or test data in pilot dataset")
        if owners.setdefault(patient, split) != split:
            raise ValueError("Patient split leakage")
        identity = (patient, split, domain)
        if pixel_owners.setdefault(row["pixel_sha256"], identity) != identity:
            raise ValueError("Identical patch across identities/domains")
        path = (root / row["file"]).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Patch path/hash validation failed")
        if row["coordinate_system"] != "original_sensor_pixels":
            raise ValueError("This version expects native sensor crops")
        if row.get("structured_fraction", 0) < .25:
            raise ValueError("Native texture curation has not passed")
        buckets.setdefault((split, domain), {}).setdefault(patient, []).append(row)
    if any(not buckets.get(key) for key in [("train", "THO"), ("train", "HE"), ("validation", "THO")]):
        raise ValueError("Missing train domains or held-out THO")
    print("DATASET_VERIFIED", len(rows), len(owners), flush=True)

    assets = output / "public_assets"
    assets.mkdir(exist_ok=True)
    provenance = {}
    for name in ("net.py", "function.py", "LICENSE"):
        url = f"https://raw.githubusercontent.com/naoto0804/pytorch-AdaIN/{UPSTREAM}/{name}"
        provenance[name] = {"url": url, "sha256": download(url, assets / name)}
    for name in ("vgg_normalised.pth", "decoder.pth"):
        url = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/" + name
        provenance[name] = {"url": url, "sha256": download(url, assets / name)}
    sys.path.insert(0, str(assets))
    spec = importlib.util.spec_from_file_location("pinned_adain_net", assets / "net.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    upstream.vgg.load_state_dict(torch.load(assets / "vgg_normalised.pth", map_location="cpu", weights_only=True))
    upstream.decoder.load_state_dict(torch.load(assets / "decoder.pth", map_location="cpu", weights_only=True))
    encoder = upstream.vgg[:31].eval().cuda().requires_grad_(False)
    decoder = upstream.decoder.cuda()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-5)

    def encode(image):
        results = []
        for index, layer in enumerate(encoder):
            image = layer(image)
            if index in (3, 10, 17, 30):
                results.append(image)
        return results

    def moments(feature):
        return feature.mean((2, 3), keepdim=True), (feature.var((2, 3), keepdim=True, unbiased=True) + 1e-5).sqrt()

    def adain(content, style):
        cm, cs = moments(content)
        sm, ss = moments(style)
        return (content - cm) / cs * ss + sm

    def tensor(row):
        with Image.open(root / row["file"]) as image:
            if image.mode != "RGB" or image.size != (256, 256):
                raise ValueError("Expected native 256x256 RGB pilot patch")
            array = np.array(image).copy()
        return torch.from_numpy(array).permute(2, 0, 1).float().div_(255).cuda()

    def sample(domain):
        patients = buckets[("train", domain)]
        return torch.stack([tensor(random.choice(patients[random.choice(sorted(patients))])) for _ in range(2)])

    # References are permanently fixed from TRAIN H&E, never from held-out target.
    references = [buckets[("train", "HE")][p][0] for p in sorted(buckets[("train", "HE")])[:3]]
    validation = []
    for patient in sorted(buckets[("validation", "THO")]):
        validation.extend(buckets[("validation", "THO")][patient][:4])
    validation = validation[:12]
    history = []

    def evaluate(step):
        decoder.eval()
        folder = output / f"step_{step:04d}"
        folder.mkdir(exist_ok=True)
        diagnostics = []
        with torch.no_grad():
            for i, row in enumerate(validation):
                content = tensor(row)[None]
                style_row = references[i % len(references)]
                style = tensor(style_row)[None]
                content_features, style_features = encode(content), encode(style)
                result = decoder(adain(content_features[-1], style_features[-1]))
                identity = decoder(content_features[-1])
                original = content[0].permute(1, 2, 0).cpu().numpy()
                predicted = result[0].permute(1, 2, 0).cpu().numpy()
                if not np.isfinite(predicted).all():
                    raise RuntimeError("Nonfinite validation output")
                # Export numerical outputs without clipping; PNG panels are display-only.
                np.save(folder / f"{i:02d}_prediction_float.npy", predicted)
                panel = Image.new("RGB", (256 * 4, 256))
                arrays = [original, predicted,
                          identity[0].permute(1, 2, 0).cpu().numpy(),
                          style[0].permute(1, 2, 0).cpu().numpy()]
                for column, array in enumerate(arrays):
                    image = Image.fromarray(np.rint(array.clip(0, 1) * 255).astype(np.uint8))
                    panel.paste(image, (256 * column, 0))
                panel.save(folder / f"{i:02d}_THO_prediction_reconstruction_STYLE.png")
                diagnostics.append({"patch": row["file"], "patient_id": row["patient_id"],
                                    "train_style_reference": style_row["file"],
                                    "identity_mae": float((identity - content).abs().mean()),
                                    "output_std": float(predicted.std()),
                                    "display_out_of_range_fraction": float(np.mean((predicted < 0) | (predicted > 1)))})
        (folder / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
        torch.save({"decoder": decoder.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "random_state": random.getstate(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "numpy_rng": np.random.get_state(), "config": vars(args)}, output / f"checkpoint_{step:04d}.pt")
        decoder.train()
        print("VALIDATED_CHECKPOINT", step, flush=True)

    run = {"upstream_commit": UPSTREAM, "assets": provenance, "config": vars(args),
           "dataset_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "interpretation": "unpaired style-transfer feasibility; no cell-level target",
           "test_used": False, "trainingAccepted": False}
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    evaluate(0)
    start = time.monotonic()
    final_step = 0
    with (output / "loss.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            if time.monotonic() - start >= args.seconds:
                break
            content, style = sample("THO"), sample("HE")
            with torch.no_grad():
                cf, sf = encode(content)[-1], encode(style)
                target = adain(cf, sf[-1])
            result = decoder(target)
            features = encode(result)
            loss_content = functional.mse_loss(features[-1], target)
            loss_style = result.new_zeros(())
            for generated, real in zip(features, sf):
                gm, gs = moments(generated)
                rm, rs = moments(real)
                loss_style = loss_style + functional.mse_loss(gm, rm) + functional.mse_loss(gs, rs)
            # Same-image reconstruction is available without paired THO/H&E targets.
            identity = decoder(cf)
            loss_identity = functional.l1_loss(identity, content)
            loss = loss_content + 10 * loss_style + 5 * loss_identity
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 10)
            optimizer.step()
            final_step = step
            row = {"step": step, "total": float(loss.detach()),
                   "content": float(loss_content.detach()), "style": float(loss_style.detach()),
                   "identity": float(loss_identity.detach()), "seconds": time.monotonic() - start,
                   "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
            log.write(json.dumps(row) + "\n")
            log.flush()
            history.append(row)
            if step % 25 == 0:
                print("TRAIN", json.dumps(row), flush=True)
            if step % 250 == 0:
                evaluate(step)
    if final_step % 250:
        evaluate(final_step)
    (output / "finished.json").write_text(json.dumps({"steps": final_step,
        "training_seconds": time.monotonic() - start, "last_metrics": history[-1] if history else None,
        "claim": "technical pilot complete; visual review and biological validation pending"}, indent=2) + "\n")
    print("PILOT_TRAIN_FINISHED", final_step, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seconds", type=int, default=2700)
    main(parser.parse_args())
