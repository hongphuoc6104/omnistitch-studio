"""DDIB-inspired latent transport with two independent small DiT score fields.

The author DDIB implementation uses different backbones. This project uses a
frozen VAE and finite positive-terminal-alpha DDIM discretization. It does not
claim exact ODE cycle consistency or anatomical preservation from cycle error.
"""
import torch

from tools.paired_stain import models


def build_score(size, assets, *, tiny=False):
    if size % 16:
        raise ValueError('Native size must be divisible by 16')
    module = models.backbone_module('C', assets)
    return module.DiT(input_size=size//8, patch_size=2, in_channels=4,
                      hidden_size=24 if tiny else 192, depth=2 if tiny else 6,
                      num_heads=3 if tiny else 6, mlp_ratio=4, class_dropout_prob=0.,
                      num_classes=1, learn_sigma=False)


def noise(model, x, t):
    return model(x, t, torch.zeros(x.shape[0], dtype=torch.long, device=x.device))


def transport(model, value, *, reverse=False, steps=50):
    """Invert a source image toward noise, or decode noise toward a domain.

    No clamping: retain numerical errors and out-of-range latents for auditing.
    A virtual clean boundary has alpha=1; the final noisy boundary stays finite.
    """
    if not 2 <= steps <= 100:
        raise ValueError('Transport step budget exceeded')
    alpha = models.schedule(value.device)
    times = torch.linspace(0, 999, steps, device=value.device).round().long().tolist()
    if reverse:
        pairs = list(zip([-1]+times[:-1], times))
    else:
        pairs = list(zip(times[::-1], times[-2::-1]+[-1]))
    x = value
    for previous, following in pairs:
        a = alpha[previous] if previous >= 0 else x.new_tensor(1.)
        b = alpha[following] if following >= 0 else x.new_tensor(1.)
        t = torch.full((x.shape[0],), max(previous, 0), dtype=torch.long, device=x.device)
        eps = noise(model, x, t)
        clean = (x-(1-a).sqrt()*eps)/a.sqrt()
        x = b.sqrt()*clean+(1-b).sqrt()*eps
        if not torch.isfinite(x).all():
            raise RuntimeError('Nonfinite DDIM bridge trajectory')
    return x
