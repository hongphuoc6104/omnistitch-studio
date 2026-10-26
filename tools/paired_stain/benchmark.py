"""GPU feasibility only: real candidate pixels, no optimizer or saved weights."""
import argparse
from pathlib import Path
import time

from tools.paired_stain import data, models
from tools.paired_stain.train import write_json


def benchmark(dataset_root, output, branch, assets, cache, steps=20):
    doc = data.validate(dataset_root, require_release=False)
    if branch not in ('B', 'C') or not 2 <= steps <= 50:
        raise ValueError('Bounded GPU benchmark requires 2–50 iterations')
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError('GPU feasibility cannot be inferred from CPU')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.cuda.reset_peak_memory_stats()
    row = next(r for r in doc['records'] if r['split'] == 'train')
    array = data.read_image(dataset_root, row['THO'], 'RGB')
    rgb = torch.from_numpy(array.copy()).permute(2, 0, 1)[None].float().cuda()/255
    size = array.shape[0]
    model = models.build(branch, size, assets).cuda()
    codec = models.load_vae('cuda', cache) if branch == 'C' else None
    condition = None
    if codec is not None:
        with torch.no_grad():
            condition = models.vae_encode(codec, rgb)
    timings = []
    for index in range(steps+3):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.autocast('cuda', dtype=torch.float16):
            if branch == 'B':
                prediction = model(rgb)
                loss = F.l1_loss(prediction, rgb)
            else:
                noise = torch.randn_like(condition)
                prediction = models.predict_noise(model, noise, condition, torch.tensor([500], device='cuda'))
                loss = F.mse_loss(prediction, noise)
        loss.backward()
        if prediction.shape != (rgb.shape if branch == 'B' else condition.shape) or not torch.isfinite(loss):
            raise RuntimeError('Invalid feasibility forward/backward')
        torch.cuda.synchronize()
        if index >= 3:
            timings.append(time.monotonic()-start)
    result = {'branch': branch, 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'microbatch': 1, 'native_size': size, 'seconds_per_forward_backward': timings,
              'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
              'parameters': sum(p.numel() for p in model.parameters()),
              'optimizer_steps': 0, 'model_weights_saved': False, 'anatomical_validation': False,
              'scope': 'hardware/gradient feasibility only, not a staining experiment',
              'dataset_sha256': data.sha256(Path(dataset_root)/'manifest.json')}
    write_json(output/'benchmark.json', result)
    print('GPU_BENCHMARK_COMPLETE', branch, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--branch', choices=['B', 'C'], required=True)
    p.add_argument('--assets', required=True)
    p.add_argument('--vae-cache')
    p.add_argument('--steps', type=int, default=20)
    args = p.parse_args()
    benchmark(args.dataset, args.output, args.branch, args.assets, args.vae_cache, args.steps)


if __name__ == '__main__':
    main()
