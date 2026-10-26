"""Two bounded nonaligned research pilots, deliberately separate from fine pairs.

B2: SwinIR + contextual/style and same-source self-similarity constraints.
C2: independently trained THO/HE DiTs + frozen VAE, DDIB-inspired DDIM transport.
Neither is an author reproduction or a claim of correct cell-level staining.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import time

from tools.nonaligned_stain import data
from tools.paired_stain import models
from tools.paired_stain.train import rng_state, restore_rng, write_json


def train(args):
    doc = data.validate(args.dataset, require_same_patient_style=args.branch == 'B2')
    if args.branch not in ('B2', 'C2') or not 1 <= args.steps <= 1000 or not 1 <= args.seconds <= 2700:
        raise ValueError('Nonaligned pilot budget exceeded')
    if not 1 <= args.accumulation <= 4:
        raise ValueError('Invalid accumulation')
    smoke = args.technical_smoke
    if smoke and (not doc.get('synthetic_fixture') or args.steps > 8 or args.seconds > 60):
        raise ValueError('CPU smoke requires a bounded synthetic fixture')
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from tools.nonaligned_stain import objectives, bridge

    if not smoke and not torch.cuda.is_available():
        raise RuntimeError('Real-data CPU training refused')
    device = torch.device('cpu' if smoke else 'cuda')
    # Repeated low-contrast features make CX gradients numerically sensitive.
    # Real T4 E2E overflowed in B2's FP16 backward before the first update.
    # Keep the whole B2 feature/model path FP32; C2 epsilon training uses AMP.
    use_amp = args.branch == 'C2' and device.type == 'cuda'
    torch.set_num_threads(1 if smoke else 2)
    random.seed(2026)
    torch.manual_seed(2026)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(2026)
        torch.cuda.reset_peak_memory_stats()
    root, output, assets = Path(args.dataset).resolve(), Path(args.output).resolve(), Path(args.assets).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    train_limit = args.seconds-min(180., max(.25, args.seconds*.2))
    size = 32 if smoke else 256
    buckets = {}
    for row in doc['records']:
        buckets.setdefault((row['split'], row['domain']), {}).setdefault(row['patient_id'], []).append(row)
    validation = []
    for patient in sorted(buckets[('validation', 'THO')]):
        validation.extend(sorted(buckets[('validation', 'THO')][patient],
                                 key=lambda r: r['structured_fraction'], reverse=True)[:4])
    validation = validation[:12]
    references = [max(buckets[('train', 'HE')][p], key=lambda r: r['structured_fraction'])
                  for p in sorted(buckets[('train', 'HE')])[:3]]

    def tensor(row):
        rgb = data.read_image(root, row, 'RGB', (size, size))
        return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().unsqueeze(0).div(255).to(device)

    def sample(domain, patient=None):
        patients = buckets[('train', domain)]
        patient = patient or random.choice(sorted(patients))
        return random.choice(patients[patient])

    networks, optimizers, scalers, ema, latents = {}, {}, {}, {}, {}
    encoder, vae = None, None
    if args.branch == 'B2':
        torch.hub.set_dir(str(assets/'torch_hub'))
        encoder = objectives.Features(fixture=smoke).to(device).eval().requires_grad_(False)
        network = models.build('B', size, assets, tiny=smoke).to(device)
        # Official scale-1 SwinIR is residual; initialize as observed input.
        torch.nn.init.zeros_(network.conv_last.weight)
        torch.nn.init.zeros_(network.conv_last.bias)
        networks['B2'] = network
    else:
        vae = models.load_vae(device, args.vae_cache) if not smoke else None
        for domain in ('THO', 'HE'):
            network = bridge.build_score(size, assets, tiny=smoke).to(device)
            networks[domain] = network
            ema[domain] = copy.deepcopy(network).eval().requires_grad_(False)
    for name, network in networks.items():
        optimizers[name] = torch.optim.AdamW(network.parameters(), lr=.0001, weight_decay=.01)
        scalers[name] = torch.amp.GradScaler('cuda', enabled=use_amp)

    def encode(image):
        if vae is not None:
            return models.vae_encode(vae, image)
        value = F.avg_pool2d(image, 8)
        return torch.cat((value, value.mean(1, keepdim=True)), 1)

    def decode(value):
        return models.vae_decode(vae, value) if vae is not None else F.interpolate(value[:, :3], scale_factor=8, mode='nearest')

    if args.branch == 'C2':
        with torch.no_grad():
            for row in doc['records']:
                if row['split'] == 'train':
                    latents[row['file']] = encode(tensor(row)).cpu()
        print('TRAIN_ONLY_LATENTS_CACHED', len(latents), flush=True)
    config = {'branch': args.branch, 'accumulation': args.accumulation, 'seed': 2026, 'size': size,
              'precision': 'fp16_amp' if use_amp else 'fp32',
              'synthetic_fixture': smoke, 'source_4X_supervision': 'unpaired; no THO-HE pixel loss',
              'SwinIR_revision': models.SWIN_REV, 'DiT_revision': models.DIT_REV,
              'VAE_revision': models.VAE_REV if vae is not None else None,
              'code_sha256': {name: data.sha256(Path(__file__).parent/name)
                              for name in ('train.py', 'data.py', 'bridge.py', 'objectives.py')},
              'backbone_loader_sha256': data.sha256(Path(models.__file__))}
    identity = {'dataset_sha256': data.sha256(root/'manifest.json'), 'config': config}
    step = 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Resume identity mismatch')
        for name in networks:
            networks[name].load_state_dict(saved['models'][name])
            optimizers[name].load_state_dict(saved['optimizers'][name])
            scalers[name].load_state_dict(saved['scalers'][name])
            if name in ema:
                ema[name].load_state_dict(saved['ema'][name])
        step = saved['step']
        if step >= args.steps:
            raise ValueError('No new updates requested')
        restore_rng(torch, saved['rng'])
    run = {**identity, 'initial_step': step, 'gpu': torch.cuda.get_device_name(0) if not smoke else None,
           'torch': torch.__version__, 'parameters': {n: sum(p.numel() for p in m.parameters()) for n, m in networks.items()},
           'test_used': False, 'paired_ground_truth_used': False, 'clinical_validation': False,
           'references_are_train_only': True, 'validation_files': [r['file'] for r in validation],
           'training_seconds_limit': train_limit, 'total_seconds_limit': args.seconds,
           'C2_transport': 'finite-alpha deterministic DDIM, 50 steps; not exact ODE' if args.branch == 'C2' else None}
    if not smoke and args.branch == 'B2':
        vgg_path = assets/'torch_hub/checkpoints/vgg19-dcbb9e9d.pth'
        run['VGG19_asset'] = {'url': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
                             'sha256': data.sha256(vgg_path), 'bytes': vgg_path.stat().st_size,
                             'normalization': 'official RGB ImageNet mean/std; relu3_1, relu4_1'}
    write_json(output/'run.json', run)

    def save_checkpoint():
        temporary = output/'checkpoint.tmp'
        torch.save({'identity': identity, 'step': step, 'models': {n: m.state_dict() for n, m in networks.items()},
                    'optimizers': {n: m.state_dict() for n, m in optimizers.items()},
                    'scalers': {n: m.state_dict() for n, m in scalers.items()},
                    'ema': {n: m.state_dict() for n, m in ema.items()}, 'rng': rng_state(torch)}, temporary)
        temporary.replace(output/'checkpoint.pt')

    def evaluate(label):
        before_rng = rng_state(torch)
        folder = output/label
        folder.mkdir()
        records = []
        for network in networks.values():
            network.eval()
        with torch.no_grad():
            for index, row in enumerate(validation):
                if time.monotonic()-start >= args.seconds:
                    break
                original = tensor(row)
                reference = tensor(references[index % len(references)])
                if args.branch == 'B2':
                    prediction = networks['B2'](original)
                    source_std = original.std((2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
                    baseline = ((original-original.mean((2, 3), keepdim=True))/source_std
                                * reference.std((2, 3), keepdim=True, unbiased=False)
                                + reference.mean((2, 3), keepdim=True))
                    panels = [('THO', original), ('prediction', prediction), ('pointwise_palette', baseline), ('HE_train_STYLE', reference)]
                    numerical = {}
                else:
                    clean = encode(original)
                    steps = 4 if smoke else 50
                    latent = bridge.transport(ema['THO'], clean, reverse=True, steps=steps)
                    stained = bridge.transport(ema['HE'], latent, steps=steps)
                    same = bridge.transport(ema['THO'], latent, steps=steps)
                    back = bridge.transport(ema['HE'], stained, reverse=True, steps=steps)
                    cycle = bridge.transport(ema['THO'], back, steps=steps)
                    prediction = decode(stained)
                    codec, same_rgb, cycle_rgb = decode(clean), decode(same), decode(cycle)
                    panels = [('THO', original), ('prediction', prediction), ('VAE_only', codec),
                              ('same_domain_solver', same_rgb), ('cycle', cycle_rgb), ('HE_train_STYLE', reference)]
                    numerical = {'same_domain_solver_latent_MAE': float((same-clean).abs().mean()),
                                 'cycle_latent_MAE_not_anatomy': float((cycle-clean).abs().mean()),
                                 'codec_same_image_MAE': float((codec-original).abs().mean()),
                                 'inverted_noise_mean': float(latent.mean()), 'inverted_noise_std': float(latent.std())}
                panel = Image.new('RGB', (size*len(panels), size))
                for column, (name, value) in enumerate(panels):
                    array = value[0].permute(1, 2, 0).float().cpu().numpy()
                    if not np.isfinite(array).all():
                        raise RuntimeError('Nonfinite validation image')
                    if name != 'HE_train_STYLE':
                        np.save(folder/f'{index:02d}_{name}.npy', array)
                    panel.paste(Image.fromarray(np.rint(np.clip(array, 0, 1)*255).astype('uint8')), (column*size, 0))
                panel.save(folder/f'{index:02d}_panels.png')
                x, y = original[0].mean(0).cpu().numpy(), prediction[0].mean(0).cpu().numpy()
                correlation = float(np.corrcoef(x.ravel(), y.ravel())[0, 1]) if x.std() and y.std() else None
                records.append({'file': row['file'], 'patient_id': row['patient_id'],
                                'source_sha256': row['sha256'], 'style_train_file': references[index % len(references)]['file'],
                                'panel_order': [name for name, _ in panels], 'grayscale_correlation_not_anatomy': correlation,
                                'prediction_clipped_fraction': float(((prediction < 0) | (prediction > 1)).float().mean()),
                                **numerical})
        for network in networks.values():
            network.train()
        restore_rng(torch, before_rng)
        write_json(folder/'metrics.json', records)
        print('NONALIGNED_EVAL', label, len(records), flush=True)
        return records

    initial = evaluate('step_0000')
    history = []
    alpha = models.schedule(device)
    with (output/'loss.jsonl').open('w') as log:
        while step < args.steps and time.monotonic()-start < train_limit:
            tick = time.monotonic()
            before_rng = rng_state(torch)
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            metrics = {}
            for _ in range(args.accumulation):
                if args.branch == 'B2':
                    row = sample('THO')
                    source, style = tensor(row), tensor(sample('HE', row['patient_id']))
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        loss, terms = objectives.b2_loss(networks['B2'], encoder, source, style)
                    terms['total'] = loss
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite B2 loss')
                    scalers['B2'].scale(loss/args.accumulation).backward()
                    for key, value in terms.items():
                        metrics[key] = metrics.get(key, 0)+float(value.detach())/args.accumulation
                else:
                    # Each model has its own domain sample, noise and optimizer;
                    # nothing treats these two samples as a matching image pair.
                    for domain in ('THO', 'HE'):
                        clean = latents[sample(domain)['file']].to(device)
                        noise = torch.randn_like(clean)
                        t = torch.randint(0, 1000, (1,), device=device)
                        a = alpha[t].view(1, 1, 1, 1)
                        noisy = a.sqrt()*clean+(1-a).sqrt()*noise
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                            loss = F.mse_loss(bridge.noise(networks[domain], noisy, t), noise)
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite C2 domain loss')
                        scalers[domain].scale(loss/args.accumulation).backward()
                        metrics[domain+'_epsilon'] = metrics.get(domain+'_epsilon', 0)+float(loss.detach())/args.accumulation
            if time.monotonic()-start >= train_limit:
                restore_rng(torch, before_rng)
                break
            # Check every model before any optimizer step for a joint boundary.
            for name in networks:
                scalers[name].unscale_(optimizers[name])
                torch.nn.utils.clip_grad_norm_(networks[name].parameters(), 1., error_if_nonfinite=True)
            for name in networks:
                scalers[name].step(optimizers[name])
                scalers[name].update()
            step += 1
            if ema:
                decay = min(.999, (1+step)/(10+step))
                with torch.no_grad():
                    for name in ema:
                        for shadow, live in zip(ema[name].parameters(), networks[name].parameters()):
                            shadow.lerp_(live, 1-decay)
            if not smoke:
                torch.cuda.synchronize()
            metrics.update({'step': step, 'seconds_per_update': time.monotonic()-tick})
            log.write(json.dumps(metrics)+'\n')
            log.flush()
            history.append(metrics)
            if step % 100 == 0:
                save_checkpoint()
                print('NONALIGNED_TRAIN', args.branch, step, json.dumps(metrics), flush=True)
    save_checkpoint()
    final = evaluate(f'step_{step:04d}_final')
    run.update({'final_step': step, 'updates_this_run': len(history), 'final_validation_predictions': len(final),
                'initial_validation_predictions': len(initial), 'wall_seconds': time.monotonic()-start,
                'peak_cuda_bytes': torch.cuda.max_memory_allocated() if not smoke else 0,
                'optimizer_updates_per_network': len(history),
                'actual_optimizer_updates_this_run': len(history)*len(networks),
                'completed': True, 'clinical_validation': False})
    write_json(output/'run.json', run)
    print('NONALIGNED_PILOT_COMPLETE', args.branch, step, flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--branch', choices=['B2', 'C2'], required=True)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--vae-cache')
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--resume')
    parser.add_argument('--technical-smoke', action='store_true')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
