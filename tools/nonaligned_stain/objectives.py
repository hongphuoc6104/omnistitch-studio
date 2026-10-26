"""Bounded contextual matching and source self-similarity; no pixel HE target.

References: Mechrez et al., ECCV2018; Kolkin et al. (STROTSS), CVPR2019.
The combination/weights and SwinIR backbone are project experiments, not an
author implementation or a proven histology objective.
"""
import torch
import torch.nn.functional as F

DEFAULT_WEIGHTS = {'CX': 1., 'source_self_similarity': 10., 'palette': 5.,
                   'same_HE_identity': 5., 'range': .1}


def validate_weights(weights=None):
    """Explicit bounded experiment coefficients, never an implicit loss change."""
    import math
    result = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if (result.keys() != DEFAULT_WEIGHTS.keys()
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100
                   for v in result.values()) or not any(result.values())):
        raise ValueError('Invalid B2 loss weights')
    return result


def tokens(feature, side=16):
    if feature.ndim != 4 or not 1 <= side <= 16:
        raise ValueError('Bounded BCHW features expected')
    side = min(side, feature.shape[-2], feature.shape[-1])
    return F.adaptive_avg_pool2d(feature.float(), (side, side)).flatten(2).transpose(1, 2)


def contextual(output, reference, bandwidth=.5):
    """Cosine CX: center by target, relative NN distances, normalized coverage."""
    x, y = tokens(output), tokens(reference)
    mean = y.mean(1, keepdim=True)
    x, y = F.normalize(x-mean, dim=-1), F.normalize(y-mean, dim=-1)
    distance = ((1-torch.bmm(x, y.transpose(1, 2)))/2).clamp_min(0)
    relative = distance/(distance.amin(-1, keepdim=True)+1e-5)
    # Softmax is the numerically stable version of exp((1-d)/h)/sum.
    weights = (-relative/bandwidth).softmax(-1)
    return -weights.amax(1).mean(-1).clamp_min(1e-8).log().mean()


def self_similarity(output, source):
    """Compare same-image relative spatial cosine distances, not HE alignment."""
    def matrix(feature):
        x = F.normalize(tokens(feature), dim=-1)
        distance = (1-torch.bmm(x, x.transpose(1, 2))).clamp_min(0)
        return distance/(distance.sum(1, keepdim=True)+1e-8)
    x, y = matrix(output), matrix(source)
    return (x-y).abs().mean()*x.shape[-1]


def palette(output, reference):
    mean = (output.mean((2, 3))-reference.mean((2, 3))).abs().mean()
    std = (output.std((2, 3), unbiased=False)-reference.std((2, 3), unbiased=False)).abs().mean()
    return mean+std


class Features(torch.nn.Module):
    """Frozen official ImageNet VGG19 with its required RGB normalization."""
    def __init__(self, *, fixture=False):
        super().__init__()
        self.fixture = fixture
        if not fixture:
            from torchvision.models import vgg19, VGG19_Weights
            self.encoder = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:21].eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, rgb):
        x = (rgb-self.mean)/self.std
        if self.fixture:
            return [F.avg_pool2d(x, 2), F.avg_pool2d(x, 4)]
        result = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in (11, 20):
                result.append(x)
        return result


def b2_terms(model, encoder, source, style):
    prediction = model(source)
    features = encoder(prediction)
    with torch.no_grad():
        content, target = encoder(source), encoder(style)
    identity_image = model(style)
    # Relative nearest-neighbour ratios need FP32 even under CUDA AMP.
    with torch.autocast(device_type=source.device.type, enabled=False):
        cx = sum(contextual(a, b) for a, b in zip(features, target))/len(features)
        structure = self_similarity(features[0], content[0])
        color = palette(prediction.float(), style.float())
        identity = F.l1_loss(identity_image.float(), style.float())
        range_penalty = (F.relu(-prediction.float())+F.relu(prediction.float()-1)).mean()
    return prediction, {'CX': cx, 'source_self_similarity': structure,
                        'palette': color, 'same_HE_identity': identity, 'range': range_penalty}


def b2_loss(model, encoder, source, style, weights=None):
    weights = validate_weights(weights)
    _, terms = b2_terms(model, encoder, source, style)
    return sum(weights[k]*v for k, v in terms.items()), terms


def gradient_profile(model, encoder, source, style, weights=None):
    """Probe TRAIN gradients without updating parameters or populating .grad.

    Output gradients exclude the separate HE identity forward by construction;
    head/shared-parameter gradients include it. Zero-init head can make the
    first shared gradient zero. Cosines with zero vectors are explicitly null.
    """
    weights = validate_weights(weights)
    prediction, terms = b2_terms(model, encoder, source, style)
    probes = [prediction, model.conv_last.weight, model.conv_first.weight]
    names = ['prediction', 'residual_head', 'shared_first_conv']
    vectors, result = {}, {}
    for key, value in terms.items():
        grads = torch.autograd.grad(value*weights[key], probes, retain_graph=True, allow_unused=True)
        vectors[key] = [torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                        for p, g in zip(probes, grads)]
        if any(not torch.isfinite(g).all() for g in vectors[key]):
            raise RuntimeError('Nonfinite diagnostic gradient')
        result[key] = {'raw_loss': float(value.detach()), 'weight': weights[key],
                       'weighted_gradient_norm': {n: float(g.norm()) for n, g in zip(names, vectors[key])}}
    cosines = {}
    keys = list(terms)
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            cosines[a+'__'+b] = {}
            for n, x, y in zip(names, vectors[a], vectors[b]):
                denominator = float(x.norm()*y.norm())
                cosines[a+'__'+b][n] = float(torch.dot(x, y))/denominator if denominator > 0 else None
    return {'terms': result, 'weighted_gradient_cosines': cosines,
            'optimizer_updates': 0, 'inputs_are_TRAIN': True}
