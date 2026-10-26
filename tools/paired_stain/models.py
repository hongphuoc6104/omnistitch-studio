"""Pinned upstream backbones, kept out of the source tree and private data.

B adapts official SwinIR to RGB scale-1 regression. C is a small project
conditional DiT with an 8-channel noisy-target/source-latent input and 4-channel
epsilon output. It is NOT the pretrained, tone-controlled D-VST reproduction.
"""
import hashlib
import importlib.util
from pathlib import Path
import urllib.request

SWIN_REV = '6545850fbf8df298df73d81f3e8cba638787c8bd'
DIT_REV = 'ed81ce2229091fd4ecc9a223645f95cf379d582b'
VAE_ID = 'stabilityai/sd-vae-ft-mse'
VAE_REV = '31f26fdeee1355a5c34592e401dd41e45d25a493'
ASSETS = {
    'swinir.py': (f'https://raw.githubusercontent.com/JingyunLiang/SwinIR/{SWIN_REV}/models/network_swinir.py',
                  '9e143898679ebeebc5d2fc94ad1b89c38aa4a4d43da4e0fcba0f93e476994913'),
    'swinir_LICENSE': (f'https://raw.githubusercontent.com/JingyunLiang/SwinIR/{SWIN_REV}/LICENSE',
                       'd66ccc11682c7458b19b376e1335e8665cbe412ff71f554bfec25f1791cddfa0'),
    'dit.py': (f'https://raw.githubusercontent.com/facebookresearch/DiT/{DIT_REV}/models.py',
               '1b8031a1340a3d1045c0bdb382334068f5f20e32edf67b3e6aba961ba91846ca'),
    'dit_LICENSE': (f'https://raw.githubusercontent.com/facebookresearch/DiT/{DIT_REV}/LICENSE.txt',
                    '7c6858d0e71dc8b858255270ad4b5004e80b5b058bf5bbc408a28430d6ddbf4b'),
}


def prepare_assets(root, branch):
    """Download only small pinned source/license files; fail on a hash change."""
    from tools.paired_stain.data import sha256
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    prefix = 'swinir' if branch == 'B' else 'dit'
    for name, (url, expected) in ASSETS.items():
        if not name.startswith(prefix):
            continue
        target = root/name
        if not target.exists():
            with urllib.request.urlopen(url, timeout=45) as stream:
                content = stream.read(1024*1024 + 1)
            if len(content) > 1024*1024 or hashlib.sha256(content).hexdigest() != expected:
                raise ValueError('Upstream asset size/hash mismatch')
            # Exclusive publication; a concurrent writer cannot be overwritten.
            with target.open('xb') as stream:
                stream.write(content)
        if sha256(target) != expected:
            raise ValueError('Corrupt pinned upstream asset')


def build(branch, size, assets, *, tiny=False):
    """Import only checksum-verified pinned code; no silent random substitute."""
    import torch
    from tools.paired_stain.data import sha256
    name = 'swinir.py' if branch == 'B' else 'dit.py'
    path = Path(assets)/name
    if sha256(path) != ASSETS[name][1]:
        raise ValueError('Backbone source hash mismatch')
    license_name = 'swinir_LICENSE' if branch == 'B' else 'dit_LICENSE'
    if sha256(Path(assets)/license_name) != ASSETS[license_name][1]:
        raise ValueError('Backbone license hash mismatch')
    spec = importlib.util.spec_from_file_location('ctup_pinned_'+branch, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if branch == 'B':
        return module.SwinIR(upscale=1, in_chans=3, img_size=size, window_size=8,
                             img_range=1., embed_dim=12 if tiny else 60,
                             depths=[2] if tiny else [2, 2, 2],
                             num_heads=[3] if tiny else [6, 6, 6], mlp_ratio=2.,
                             upsampler='', resi_connection='1conv', drop_path_rate=0.)
    if branch != 'C' or size % 16:
        raise ValueError('Conditional latent DiT needs size divisible by 16')
    model = module.DiT(input_size=size//8, patch_size=2, in_channels=8,
                       hidden_size=24 if tiny else 192, depth=2 if tiny else 6,
                       num_heads=3 if tiny else 6, mlp_ratio=4,
                       class_dropout_prob=0., num_classes=1, learn_sigma=False)
    # The official class couples in/out channels. The project conditions on
    # concat([noisy target, source]) while predicting ONLY four target channels.
    model.out_channels = 4
    model.final_layer.linear = torch.nn.Linear(model.pos_embed.shape[-1], 2*2*4)
    torch.nn.init.zeros_(model.final_layer.linear.weight)
    torch.nn.init.zeros_(model.final_layer.linear.bias)
    return model


def predict_noise(model, noisy, source, timestep):
    import torch
    labels = torch.zeros(noisy.shape[0], device=noisy.device, dtype=torch.long)
    return model(torch.cat((noisy, source), dim=1), timestep, labels)


def load_vae(device, cache=None, *, local_only=True):
    """Use pinned public VAE, never fit the codec to validation/test images."""
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(VAE_ID, revision=VAE_REV, cache_dir=cache,
                                         local_files_only=local_only,
                                         use_safetensors=True).to(device).eval().requires_grad_(False)


def vae_encode(vae, rgb):
    # Deterministic posterior mean allows reproducible codec diagnostics and
    # keeps source-conditioning noise separate from target diffusion noise.
    return vae.encode(rgb*2-1).latent_dist.mode() * vae.config.scaling_factor


def vae_decode(vae, latent):
    return (vae.decode(latent/vae.config.scaling_factor).sample+1)/2


def schedule(device):
    import torch
    return torch.cumprod(1-torch.linspace(.0001, .02, 1000, device=device), dim=0)


def ddim(model, source, generator, steps=50):
    """Deterministic DDIM (eta=0), epsilon parameterization, no latent clamp."""
    import torch
    alpha = schedule(source.device)
    x = torch.randn(source.shape, generator=generator, device=source.device)
    times = torch.linspace(999, 0, steps, device=source.device).round().long()
    for index, t in enumerate(times):
        eps = predict_noise(model, x, source, t.expand(source.shape[0]))
        a = alpha[t]
        previous = alpha[times[index+1]] if index+1 < len(times) else x.new_tensor(1.)
        clean = (x - (1-a).sqrt()*eps)/a.sqrt()
        x = previous.sqrt()*clean + (1-previous).sqrt()*eps
    return x
