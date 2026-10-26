"""Prepare pinned assets or inspect the frozen codec without training a model."""
import argparse
import json
from pathlib import Path
import time

from tools.paired_stain import data, models
from tools.paired_stain.train import write_json


def assets(root, branch, cache=None):
    models.prepare_assets(root, branch)
    if cache and branch == 'C':
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(models.VAE_ID, revision=models.VAE_REV, cache_dir=cache,
                                         allow_patterns=['config.json', 'diffusion_pytorch_model.safetensors']))
        write_json(Path(root)/'VAE_assets.json', {
            'model': models.VAE_ID, 'revision': models.VAE_REV,
            'files': {p.name: data.sha256(p) for p in snapshot.iterdir() if p.is_file()}})


def codec(dataset_root, output, cache, device, limit, seconds):
    """Round-trip same-image reconstruction; no train, no automatic approval.

    Same-image MAE/gradient error is a codec diagnostic, not a focus measurement
    or THO–H&E accuracy. Both PNGs and unclipped numerical outputs are retained.
    """
    doc = data.validate(dataset_root, require_release=False)
    if device not in ('cpu', 'cuda') or not 1 <= limit <= 512 or not 1 <= seconds <= 1800:
        raise ValueError('Codec diagnostic exceeds its bounded budget')
    import numpy as np
    from PIL import Image
    import torch

    torch.set_num_threads(2)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested codec CUDA is unavailable')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    vae = models.load_vae(device, cache)
    records, seen = [], set()
    # Stable stratification prevents an early limit from selecting only THO or
    # only train. Repeated windows stay correlated; this is not a sample size.
    ordered = sorted(doc['records'], key=lambda r: (r['patient_id'], r['id']))
    buckets = [[(r, domain) for r in ordered if r['split'] == split for domain in [d]]
               for split in ('train', 'validation') for d in ('THO', 'HE')]
    queue = [item for index in range(max(map(len, buckets))) for bucket in buckets for item in bucket[index:index+1]]
    with torch.no_grad():
        for row, domain in queue:
            if len(records) >= limit or time.monotonic()-start >= seconds:
                break
            item = row[domain]
            if item['sha256'] in seen:
                continue
            seen.add(item['sha256'])
            original = data.read_image(dataset_root, item, 'RGB')
            rgb = torch.from_numpy(original.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
            reconstructed = models.vae_decode(vae, models.vae_encode(vae, rgb))
            if reconstructed.shape != rgb.shape or not torch.isfinite(reconstructed).all():
                raise RuntimeError('Invalid frozen codec reconstruction')
            array = reconstructed[0].permute(1, 2, 0).cpu().numpy()
            index = len(records)
            np.save(output/f'{index:03d}_reconstruction.npy', array)
            displayed = Image.fromarray(np.rint(np.clip(array, 0, 1)*255).astype('uint8'))
            reconstruction = output/f'{index:03d}_reconstruction.png'
            displayed.save(reconstruction)
            panel = Image.new('RGB', (original.shape[1]*2, original.shape[0]))
            panel.paste(Image.fromarray(original), (0, 0))
            panel.paste(displayed, (original.shape[1], 0))
            panel.save(output/f'{index:03d}_{domain}_original_reconstruction.png')
            records.append({'id': row['id'], 'patient_id': row['patient_id'], 'split': row['split'],
                            'domain': domain, 'input_sha256': item['sha256'],
                            'same_image_MAE': float((reconstructed-rgb).abs().mean()),
                            'display_clipped_fraction': float(np.mean((array < 0) | (array > 1))),
                            'reconstruction_file': reconstruction.name,
                            'reconstruction_sha256': data.sha256(reconstruction)})
            print('CODEC_PROGRESS', len(records), domain, flush=True)
    report = {'VAE_id': models.VAE_ID, 'VAE_revision': models.VAE_REV, 'synthetic_codec': False,
              'dataset_manifest_sha256': data.sha256(Path(dataset_root)/'manifest.json'),
              'device': device, 'wall_seconds': time.monotonic()-start, 'records': records,
              'automatic_approval': False, 'training_started': False,
              'complete_unique_image_coverage': len(seen) == len({r[d]['sha256'] for r in doc['records'] for d in ('THO', 'HE')})}
    write_json(output/'report.json', report)
    write_json(output/'approval_PENDING.json', {'approved': False, 'reviewer': None,
                                               'review_evidence': None, 'report_file': 'report.json',
                                               'report_sha256': data.sha256(output/'report.json')})
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('assets')
    a.add_argument('--assets', required=True)
    a.add_argument('--branch', choices=['B', 'C'], required=True)
    a.add_argument('--vae-cache')
    c = sub.add_parser('codec')
    c.add_argument('--dataset', required=True)
    c.add_argument('--output', required=True)
    c.add_argument('--vae-cache', required=True)
    c.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    c.add_argument('--limit', type=int, default=16)
    c.add_argument('--seconds', type=int, default=600)
    args = p.parse_args()
    if args.command == 'assets':
        assets(args.assets, args.branch, args.vae_cache)
        print('PINNED_ASSETS_READY')
    else:
        report = codec(args.dataset, args.output, args.vae_cache, args.device, args.limit, args.seconds)
        print('CODEC_DIAGNOSTIC_COMPLETE', len(report['records']))


if __name__ == '__main__':
    main()
