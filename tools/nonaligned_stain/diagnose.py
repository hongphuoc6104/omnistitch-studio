"""Bounded checkpoint-only evaluation: no optimizer, no paired HE targets.

Private pickle checkpoints require an explicitly supplied, verified SHA256.
Validation is never added to a training cache. Image outputs are native size;
unclipped floats and clipped viewing PNGs are retained separately.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from tools.nonaligned_stain import data
from tools.paired_stain import models
from tools.paired_stain.train import write_json


def diagnose(args):
    doc = data.validate(args.dataset)
    checkpoint = Path(args.checkpoint)
    if (not 0 < checkpoint.stat().st_size <= 512*1024**2
            or data.sha256(checkpoint) != args.checkpoint_sha256):
        raise ValueError('Untrusted checkpoint size/SHA256')
    if not 1 <= args.seconds <= 2700:
        raise ValueError('Diagnostic budget exceeded')
    smoke = getattr(args, 'technical_smoke', False)
    if smoke and (not doc.get('synthetic_fixture') or args.seconds > 60):
        raise ValueError('Synthetic diagnostic requires a fixture and bounded budget')
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from tools.nonaligned_stain import bridge, objectives

    if not smoke and not torch.cuda.is_available():
        raise RuntimeError('Real-data CPU diagnostic refused')
    torch.set_num_threads(1 if smoke else 2)
    device = torch.device('cpu' if smoke else 'cuda')
    start = time.monotonic()
    if not smoke:
        torch.cuda.reset_peak_memory_stats()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = saved['identity']['config']
    size = 32 if smoke else 256
    if (saved['identity']['dataset_sha256'] != data.sha256(Path(args.dataset)/'manifest.json')
            or config['branch'] != args.branch or config['size'] != size
            or config['synthetic_fixture'] != bool(doc.get('synthetic_fixture'))
            or config['DiT_revision'] != models.DIT_REV or config['SwinIR_revision'] != models.SWIN_REV
            or (args.branch == 'C2' and not smoke and config['VAE_revision'] != models.VAE_REV)):
        raise ValueError('Checkpoint/dataset/backbone identity mismatch')
    root, out, assets = Path(args.dataset), Path(args.output), Path(args.assets)
    out.mkdir(parents=True, exist_ok=False)
    status = {'branch': args.branch, 'checkpoint_sha256': args.checkpoint_sha256,
              'checkpoint_step': saved['step'], 'checkpoint_identity': saved['identity'],
              'evaluation_code_sha256': data.sha256(Path(__file__)),
              'evaluation_dependencies_sha256': {n: data.sha256(Path(__file__).with_name(n))
                                               for n in ('objectives.py', 'bridge.py', 'data.py')},
              'backbone_loader_sha256': data.sha256(Path(models.__file__)),
              'optimizer_updates': 0, 'test_used': False, 'paired_ground_truth': False,
              'synthetic_fixture': smoke, 'evaluation_precision': 'fp32',
              'GPU': torch.cuda.get_device_name(0) if not smoke else None,
              'completed': False}
    records, by_patient = [], defaultdict(list)

    def tick(stage):
        status.update(stage=stage, wall_seconds=time.monotonic()-start)
        write_json(out/'run.json', status)
        if time.monotonic()-start >= args.seconds:
            raise TimeoutError('Diagnostic deadline; partial outputs are not full validation')

    def tensor(row):
        array = data.read_image(root, row, 'RGB', (size, size))
        return torch.from_numpy(array.copy()).permute(2, 0, 1).float()[None].to(device)/255

    def image(folder, name, value):
        folder = out/folder
        folder.mkdir(exist_ok=True)
        array = value.detach()[0].permute(1, 2, 0).float().cpu().numpy()
        if array.shape != (size, size, 3) or not np.isfinite(array).all():
            raise RuntimeError('Nonfinite or resized diagnostic image')
        np.save(folder/(name+'.npy'), array)
        Image.fromarray(np.rint(np.clip(array, 0, 1)*255).astype('uint8')).save(folder/(name+'.png'))
        return float(((value < 0) | (value > 1)).float().mean())

    validation = sorted([r for r in doc['records'] if r['split'] == 'validation'],
                        key=lambda r: (r['domain'], r['patient_id'], -r['structured_fraction'], r['file']))
    sources = [r for r in validation if r['domain'] == 'THO']
    train_he = sorted([r for r in doc['records'] if r['split'] == 'train' and r['domain'] == 'HE'],
                      key=lambda r: (r['patient_id'], -r['structured_fraction'], r['file']))
    references = []
    for patient in sorted({r['patient_id'] for r in train_he})[:3]:
        references.append(next(r for r in train_he if r['patient_id'] == patient))
    quick = [r for p in sorted({r['patient_id'] for r in sources})
             for r in [s for s in sources if s['patient_id'] == p][:4]][:12]
    quick_indices = {r['file']: i for i, r in enumerate(quick)}
    networks, vae, encoder = {}, None, None
    if args.branch == 'B2':
        network = models.build('B', size, assets, tiny=smoke).to(device)
        network.load_state_dict(saved['models']['B2'], strict=True)
        networks['B2'] = network.eval().requires_grad_(False)
    else:
        vae = models.load_vae(device, args.vae_cache) if not smoke else None
        for domain in ('THO', 'HE'):
            network = bridge.build_score(size, assets, tiny=smoke).to(device)
            network.load_state_dict(saved['ema'][domain], strict=True)
            networks[domain] = network.eval().requires_grad_(False)
    del saved

    def encode(rgb):
        if vae is not None:
            return models.vae_encode(vae, rgb)
        z = F.avg_pool2d(rgb, 8)
        return torch.cat((z, z.mean(1, keepdim=True)), 1)

    def decode(z):
        return models.vae_decode(vae, z) if vae is not None else F.interpolate(z[:, :3], scale_factor=8, mode='nearest')

    try:
        with torch.no_grad():
            tick('full_validation_translation')
            for i, row in enumerate(sources):
                tick('full_validation_translation')
                source = tensor(row)
                if args.branch == 'B2':
                    prediction = networks['B2'](source)
                else:
                    z = encode(source)
                    latent = bridge.transport(networks['THO'], z, reverse=True, steps=4 if smoke else 50)
                    prediction = decode(bridge.transport(networks['HE'], latent, steps=4 if smoke else 50))
                reference_row = references[quick_indices.get(row['file'], i) % len(references)]
                reference = tensor(reference_row)
                source_std = source.std((2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
                baseline = ((source-source.mean((2, 3), keepdim=True))/source_std
                            * reference.std((2, 3), keepdim=True, unbiased=False)
                            + reference.mean((2, 3), keepdim=True))
                name = f'{i:03d}'
                clipped = image('predictions', name, prediction)
                image('source', name, source)
                image('pointwise', name, baseline)
                image('TRAIN_STYLE', name, reference)
                record = {'file': row['file'], 'source_sha256': row['sha256'], 'patient_id': row['patient_id'],
                          'index': name, 'style_file': reference_row['file'], 'style_split': 'train',
                          'source_MAE_not_accuracy': float((prediction-source).abs().mean()),
                          'palette_not_accuracy': float(objectives.palette(prediction, reference)),
                          'out_of_range_RGB_elements': clipped}
                records.append(record)
                by_patient[row['patient_id']].append(record)
                write_json(out/'translation.json', records)

            if args.branch == 'C2':
                codec, denoise, controls = [], [], []
                alpha = models.schedule(device)
                train_diagnostic = []
                for domain in ('THO', 'HE'):
                    for patient in sorted({r['patient_id'] for r in doc['records'] if r['split'] == 'train' and r['domain'] == domain}):
                        train_diagnostic.extend(sorted([r for r in doc['records'] if r['split'] == 'train' and
                                                        r['domain'] == domain and r['patient_id'] == patient],
                                                       key=lambda r: r['file'])[:4])
                # No validation cache is shared with any training process.
                for i, row in enumerate(validation+train_diagnostic):
                    tick('codec_and_fixed_noise_denoising')
                    rgb = tensor(row)
                    z = encode(rgb)
                    if row['split'] == 'validation':
                        reconstructed = decode(z)
                        image('codec', f'{i:03d}', reconstructed)
                        codec.append({'file': row['file'], 'patient_id': row['patient_id'], 'domain': row['domain'],
                                      'same_image_MAE': float((reconstructed-rgb).abs().mean()),
                                      'latent_mean_channels': z.mean((0, 2, 3)).cpu().tolist(),
                                      'latent_std_channels': z.std((0, 2, 3), unbiased=False).cpu().tolist()})
                    generator = torch.Generator(device=device).manual_seed(7300+i)
                    for repetition in range(2):
                        eps = torch.randn(z.shape, device=device, generator=generator)
                        for t in ([0, 500, 999] if smoke else [0, 50, 200, 500, 800, 999]):
                            a = alpha[t]
                            noisy = a.sqrt()*z+(1-a).sqrt()*eps
                            predicted = bridge.noise(networks[row['domain']], noisy, torch.tensor([t], device=device))
                            mse, zero = float(F.mse_loss(predicted, eps)), float(eps.square().mean())
                            if not np.isfinite(mse):
                                raise RuntimeError('Nonfinite denoising diagnostic')
                            denoise.append({'file': row['file'], 'source_sha256': row['sha256'], 'patient_id': row['patient_id'],
                                            'split': row['split'], 'domain': row['domain'], 't': t, 'repetition': repetition,
                                            'epsilon_MSE': mse, 'zero_predictor_MSE': zero, 'ratio_to_zero': mse/zero})
                write_json(out/'codec.json', codec)
                write_json(out/'denoising.json', denoise)
                for i, row in enumerate(quick):
                    tick('solver_controls')
                    z = encode(tensor(row))
                    for steps in ([2, 4] if smoke else [25, 50, 100]):
                        tick('solver_controls')
                        latent = bridge.transport(networks['THO'], z, reverse=True, steps=steps)
                        stained = bridge.transport(networks['HE'], latent, steps=steps)
                        same = bridge.transport(networks['THO'], latent, steps=steps)
                        back = bridge.transport(networks['HE'], stained, reverse=True, steps=steps)
                        cycle = bridge.transport(networks['THO'], back, steps=steps)
                        prefix = f'{i:02d}_{steps}'
                        for name, value in [('same', same), ('cross', stained), ('cycle', cycle)]:
                            image('solver', prefix+'_'+name, decode(value))
                        controls.append({'file': row['file'], 'patient_id': row['patient_id'], 'steps_each_direction': steps,
                                         'same_latent_MAE': float((same-z).abs().mean()),
                                         'cycle_latent_MAE_not_anatomy': float((cycle-z).abs().mean()),
                                         'inverted_mean': float(latent.mean()), 'inverted_std': float(latent.std())})
                        write_json(out/'solver.json', controls)
                for domain in ('THO', 'HE'):
                    for seed in range(4100, 4102 if smoke else 4116):
                        tick('domain_only_generation')
                        generator = torch.Generator(device=device).manual_seed(seed)
                        latent = torch.randn((1, 4, size//8, size//8), generator=generator, device=device)
                        prediction = decode(bridge.transport(networks[domain], latent, steps=4 if smoke else 50))
                        image('domain_samples', domain+'_'+str(seed), prediction)
                status.update(codec_records=len(codec), denoising_records=len(denoise), solver_records=len(controls),
                              terminal_alpha=float(alpha[-1]),
                              generation_warning='Pure Gaussian at positive terminal alpha is an approximate marginal control, not a proof of correct density.')
        summary = {}
        for patient, rows in by_patient.items():
            summary[patient] = {key: float(np.median([r[key] for r in rows]))
                                for key in ('source_MAE_not_accuracy', 'palette_not_accuracy', 'out_of_range_RGB_elements')}
            summary[patient]['THO_predictions'] = len(rows)
        write_json(out/'by_patient.json', summary)
        if len(records) != len(sources):
            raise RuntimeError('Incomplete THO validation')
        status.update(completed=True, stage='completed', THO_predictions=len(records),
                      expected_THO_predictions=len(sources), validation_images=len(validation),
                      wall_seconds=time.monotonic()-start,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated() if not smoke else 0)
        write_json(out/'run.json', status)
        print('NONALIGNED_DIAGNOSTIC_COMPLETE', args.branch, len(records), flush=True)
        return status
    except BaseException as error:
        status.update(stage='failed', error=type(error).__name__+': '+str(error), THO_predictions=len(records),
                      wall_seconds=time.monotonic()-start)
        write_json(out/'run.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'checkpoint', 'checkpoint-sha256', 'output', 'assets'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--branch', choices=['B2', 'C2'], required=True)
    parser.add_argument('--vae-cache')
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--technical-smoke', action='store_true')
    diagnose(parser.parse_args())
