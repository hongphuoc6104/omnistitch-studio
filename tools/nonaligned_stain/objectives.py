"""Bounded contextual matching and source self-similarity; no pixel HE target.

References: Mechrez et al., ECCV2018; Kolkin et al. (STROTSS), CVPR2019.
The combination/weights and SwinIR backbone are project experiments, not an
author implementation or a proven histology objective.
"""
import torch
import torch.nn.functional as F


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


def b2_loss(model, encoder, source, style):
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
    total = cx+10*structure+5*color+5*identity+.1*range_penalty
    return total, {'CX': cx, 'source_self_similarity': structure,
                   'palette': color, 'same_HE_identity': identity, 'range': range_penalty}
