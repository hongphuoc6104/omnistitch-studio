"""Bounded B/C research pilots with strict release/codec gates and resume.

Use `python -m tools.paired_stain.train --help`. No real-data CPU training,
silent codec substitute, test-set ingestion or candidate-to-positive promotion.
"""
import argparse
import json
from pathlib import Path
import random
import time

from tools.paired_stain import data, models


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def required_codec(dataset, approval_file):
    """Fine-scale latent training needs a separately reviewed round-trip report."""
    if not approval_file:
        raise ValueError('C requires reviewed real-image VAE reconstruction evidence')
    path = Path(approval_file)
    approval = json.loads(path.read_text())
    report_path = data.contained(path.parent, approval['report_file'])
    if data.sha256(report_path) != approval['report_sha256']:
        raise ValueError('Codec report checksum mismatch')
    report = json.loads(report_path.read_text())
    needed = {r[k]['sha256'] for r in dataset['records'] for k in ('THO', 'HE')}
    seen = {r['input_sha256'] for r in report['records']}
    if (approval.get('approved') is not True or not approval.get('reviewer')
            or not approval.get('review_evidence') or report.get('VAE_revision') != models.VAE_REV
            or report.get('synthetic_codec') is not False or not needed.issubset(seen)):
        raise ValueError('Codec approval/coverage is incomplete')
    for row in report['records']:
        prediction = data.contained(report_path.parent, row['reconstruction_file'])
        if data.sha256(prediction) != row['reconstruction_sha256']:
            raise ValueError('Codec reconstruction changed after review')
    return {'approval_sha256': data.sha256(path), 'report_sha256': data.sha256(report_path)}


def rng_state(torch):
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(torch, state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def train(args):
    """Train one independent bounded job, returning quantitative run metadata."""
    # All scientific gates precede GPU/weights/model allocation.
    dataset = data.validate(args.dataset)
    if args.branch not in ('B', 'C') or not 1 <= args.steps <= 1000 or not 1 <= args.seconds <= 2700:
        raise ValueError('Pilot is bounded to 1000 updates / 45 minutes')
    if not 1 <= args.accumulation <= 8 or not 0 < args.lr <= .001:
        raise ValueError('Invalid microbatch accumulation/learning rate')
    smoke = args.technical_smoke
    if smoke and (not dataset.get('synthetic_fixture') or args.steps > 8 or args.seconds > 60):
        raise ValueError('CPU smoke is restricted to bounded synthetic fixtures')
    codec_evidence = required_codec(dataset, args.codec_approval) if args.branch == 'C' and not smoke else None
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F

    if not smoke and not torch.cuda.is_available():
        raise RuntimeError('Real-data pilot requires CUDA; CPU training refused')
    torch.set_num_threads(1 if smoke else 2)
    device = torch.device('cpu' if smoke else 'cuda')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    root = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    size = data.read_image(root, dataset['records'][0]['THO'], 'RGB').shape[0]
    model = models.build(args.branch, size, args.assets, tiny=smoke).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    vae = models.load_vae(device, args.vae_cache) if args.branch == 'C' and not smoke else None
    alpha = models.schedule(device) if args.branch == 'C' else None
    dataset_hash = data.sha256(root/'manifest.json')
    config = {'branch': args.branch, 'seed': args.seed, 'accumulation': args.accumulation,
              'lr': args.lr, 'size': size, 'technical_smoke': smoke,
              'runner_files_sha256': {name: data.sha256(Path(__file__).parent/name)
                                      for name in ('train.py', 'data.py', 'models.py')},
              'upstream': models.SWIN_REV if args.branch == 'B' else models.DIT_REV,
              'VAE_revision': models.VAE_REV if vae is not None else None,
              'codec_evidence': codec_evidence}
    identity = {'config': config, 'dataset_sha256': dataset_hash}
    step = 0
    if args.resume:
        # Resume only a trusted private checkpoint produced by this runner.
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if checkpoint['identity'] != identity:
            raise ValueError('Resume dataset/model/config differs')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        step = checkpoint['step']
        if step >= args.steps:
            raise ValueError('Resume target must exceed completed step')
        restore_rng(torch, checkpoint['rng'])
    buckets = {}
    for row in dataset['records']:
        buckets.setdefault(row['split'], {}).setdefault(row['patient_id'], []).append(row)
    validation = [row for p in sorted(buckets['validation']) for row in sorted(buckets['validation'][p], key=lambda r: r['id'])[:4]][:12]
    run = {**identity, 'model_parameters': sum(p.numel() for p in model.parameters()),
           'device': str(device), 'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
           'torch': torch.__version__, 'test_used': False, 'synthetic_codec': smoke and args.branch == 'C',
           'scientific_training_result': False, 'records': len(dataset['records']),
           'validation_ids': [r['id'] for r in validation], 'initial_step': step,
           'scope': 'project adaptation; technical feasibility, not validated virtual staining'}
    write_json(output/'run.json', run)

    def tensor(row, key):
        image = data.read_image(root, row[key], 'RGB')
        return torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)

    def encode(rgb):
        if vae is not None:
            return models.vae_encode(vae, rgb)
        # Explicit synthetic-only fixture for loop tests; never a real VAE.
        rgb = F.avg_pool2d(rgb, 8)
        return torch.cat((rgb, rgb.mean(1, keepdim=True)), dim=1)

    def decode(latent):
        return models.vae_decode(vae, latent) if vae is not None else F.interpolate(latent[:, :3], scale_factor=8, mode='nearest')

    def checkpoint_now():
        temporary = output/'checkpoint.tmp'
        torch.save({'identity': identity, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(), 'step': step, 'rng': rng_state(torch)}, temporary)
        temporary.replace(output/'checkpoint.pt')

    def evaluate():
        model.eval()
        records = []
        folder = output/'validation'
        folder.mkdir(exist_ok=True)
        saved_rng = rng_state(torch)
        with torch.no_grad():
            for index, row in enumerate(validation):
                if time.monotonic()-start >= args.seconds:
                    break
                source, target = tensor(row, 'THO'), tensor(row, 'HE')
                seeds = [2026] if args.branch == 'B' else [2026, 2027, 2028]
                conditioning = encode(source) if args.branch == 'C' else None
                for seed in seeds:
                    if args.branch == 'B':
                        prediction = model(source)
                    else:
                        generator = torch.Generator(device=device).manual_seed(seed)
                        prediction = decode(models.ddim(model, conditioning, generator, steps=4 if smoke else 50))
                    if prediction.shape != source.shape or not torch.isfinite(prediction).all():
                        raise RuntimeError('Invalid validation prediction')
                    array = prediction[0].permute(1, 2, 0).cpu().numpy()
                    stem = f'{index:03d}_seed{seed}'
                    np.save(folder/(stem+'.npy'), array)
                    panel = Image.new('RGB', (size*3, size))
                    for column, value in enumerate((source, prediction, target)):
                        displayed = value[0].permute(1, 2, 0).cpu().numpy()
                        panel.paste(Image.fromarray(np.rint(np.clip(displayed, 0, 1)*255).astype('uint8')), (column*size, 0))
                    panel.save(folder/(stem+'.png'))
                    records.append({'id': row['id'], 'patient_id': row['patient_id'], 'seed': seed,
                                    'paired_L1_diagnostic_not_cell_accuracy': float((prediction-target).abs().mean()),
                                    'display_clipped_fraction': float(np.mean((array < 0) | (array > 1)))})
        restore_rng(torch, saved_rng)
        write_json(folder/'metrics.json', records)
        model.train()
        return len(records)

    durations, losses = [], []
    with (output/'loss.jsonl').open('w') as log:
        while step < args.steps and time.monotonic()-start < args.seconds:
            before = time.monotonic()
            before_rng = rng_state(torch)
            optimizer.zero_grad(set_to_none=True)
            value = 0.
            for _ in range(args.accumulation):
                patients = buckets['train']
                patient = random.choice(sorted(patients))
                row = random.choice(patients[patient])
                source, target = tensor(row, 'THO'), tensor(row, 'HE')
                if args.branch == 'C':
                    with torch.no_grad():
                        condition, clean = encode(source), encode(target)
                    t = torch.randint(0, 1000, (1,), device=device)
                    noise = torch.randn_like(clean)
                    a = alpha[t].view(-1, 1, 1, 1)
                    noisy = a.sqrt()*clean + (1-a).sqrt()*noise
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    if args.branch == 'B':
                        loss = F.l1_loss(model(source), target)
                    else:
                        loss = F.mse_loss(models.predict_noise(model, noisy, condition, t), noise)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                scaler.scale(loss/args.accumulation).backward()
                value += float(loss.detach())/args.accumulation
            if time.monotonic()-start >= args.seconds:
                optimizer.zero_grad(set_to_none=True)
                restore_rng(torch, before_rng)
                break
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            step += 1
            elapsed = time.monotonic()-before
            durations.append(elapsed)
            losses.append(value)
            log.write(json.dumps({'step': step, 'loss': value, 'gradient_norm': float(norm),
                                  'seconds_per_update': elapsed, 'elapsed_seconds': time.monotonic()-start})+'\n')
            log.flush()
            if step % 25 == 0:
                checkpoint_now()
                print('PAIRED_TRAIN', step, round(value, 6), flush=True)
    checkpoint_now()
    run.update({'final_step': step, 'losses': losses, 'updates_this_run': len(durations),
                'seconds_per_update': durations,
                'peak_cuda_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
                'validation_predictions': evaluate(), 'wall_seconds': time.monotonic()-start,
                'scientific_training_result': False, 'completed': True})
    write_json(output/'run.json', run)
    print('PAIRED_PILOT_COMPLETE', step, flush=True)
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--branch', choices=['B', 'C'], required=True)
    p.add_argument('--assets', required=True)
    p.add_argument('--vae-cache')
    p.add_argument('--codec-approval')
    p.add_argument('--resume')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--seconds', type=int, default=2700)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--lr', type=float, default=.0001)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--technical-smoke', action='store_true')
    train(p.parse_args())


if __name__ == '__main__':
    main()
