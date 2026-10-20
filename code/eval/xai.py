"""Saliency generation and faithfulness/localisation metrics (RQ4 core).

Faithfulness asks: do the highlighted regions reflect what the model
ACTUALLY used? Measured by deletion/insertion (Petsiuk et al., RISE).
Localisation asks: do they fall on the lesion? Measured by IoU and the
pointing game against HAM10000 lesion masks.

COST NOTE (this drives the whole compute budget): deletion+insertion needs
~2*steps forward passes per image. At steps=32 that is 64 extra passes per
image per explainer -- which is why the Week-2 gate measures throughput
before the full campaign is scheduled.
"""
import numpy as np
import torch
import torch.nn.functional as F


def target_layer(model, arch):
    """Last conv stage, where spatial detail still exists."""
    if arch == "resnet50":
        return model.layer4[-1]
    if arch == "efficientnet_b3":
        return model.conv_head
    if arch == "densenet121":
        return model.features[-1]
    raise ValueError(f"no target layer registered for {arch!r}")


class GradCAM:
    """Grad-CAM for a SINGLE-LOGIT (binary) head.

    Implemented directly rather than taken from a library because the binary
    case needs the gradient of the melanoma logit itself, not of a class
    index chosen by argmax -- with one output, argmax is always 0 and the
    library default would explain the wrong quantity.
    """

    def __init__(self, model, layer):
        self.model, self.layer = model, layer
        self.acts = self.grads = None
        self._h = [layer.register_forward_hook(self._fwd),
                   layer.register_full_backward_hook(self._bwd)]

    def _fwd(self, m, i, o):
        self.acts = o

    def _bwd(self, m, gi, go):
        self.grads = go[0]

    def remove(self):
        for h in self._h:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.remove()

    def __call__(self, x):
        """x: (B,3,H,W) -> (B,H,W) saliency in [0,1]."""
        self.model.zero_grad(set_to_none=True)
        # autocast is deliberately NOT used: fp16 gradients at this layer
        # underflow to zero and silently yield all-zero maps.
        with torch.enable_grad():
            logit = self.model(x).squeeze(1)
            logit.sum().backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)      # GAP over spatial
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                            align_corners=False).squeeze(1)
        # normalise per image
        b = cam.shape[0]
        mn = cam.view(b, -1).min(1).values.view(b, 1, 1)
        mx = cam.view(b, -1).max(1).values.view(b, 1, 1)
        return ((cam - mn) / (mx - mn + 1e-8)).detach()


@torch.no_grad()
def _probs(model, x):
    return torch.sigmoid(model(x).squeeze(1).float())


@torch.no_grad()
def deletion_insertion(model, x, sal, steps=32, baseline="blur"):
    """Deletion and insertion AUC (RISE).

    deletion  -- progressively REMOVE the most salient pixels; a faithful
                 explanation makes the melanoma probability fall FAST, so
                 LOW deletion AUC is good.
    insertion -- progressively ADD the most salient pixels to a baseline;
                 a faithful explanation makes probability rise fast, so
                 HIGH insertion AUC is good.

    The baseline is a blurred copy rather than zeros/black: a black patch is
    far outside the data distribution and produces spuriously large drops
    that look like faithfulness but are just distribution shift.

    Returns (del_auc, ins_auc), each averaged over the batch.
    """
    B, C, H, W = x.shape
    n = H * W
    if baseline == "blur":
        k = max(3, (min(H, W) // 8) * 2 + 1)
        base = _gauss_blur(x, k)
    else:
        base = torch.zeros_like(x)

    order = sal.view(B, -1).argsort(dim=1, descending=True)
    chunk = max(1, n // steps)

    del_curve, ins_curve = [], []
    xd, xi = x.clone(), base.clone()
    del_curve.append(_probs(model, xd))
    ins_curve.append(_probs(model, xi))
    flat_x = x.view(B, C, -1)
    flat_b = base.view(B, C, -1)
    for s in range(steps):
        idx = order[:, s * chunk:(s + 1) * chunk]
        if idx.numel() == 0:
            break
        ii = idx.unsqueeze(1).expand(B, C, idx.shape[1])
        xd.view(B, C, -1).scatter_(2, ii, flat_b.gather(2, ii))
        xi.view(B, C, -1).scatter_(2, ii, flat_x.gather(2, ii))
        del_curve.append(_probs(model, xd))
        ins_curve.append(_probs(model, xi))
    d = torch.stack(del_curve, 1).mean(1)     # trapezoid ~ mean for uniform steps
    i = torch.stack(ins_curve, 1).mean(1)
    return d.cpu().numpy(), i.cpu().numpy()


def _gauss_blur(x, k):
    sig = k / 6.0
    c = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(c ** 2) / (2 * sig ** 2))
    g = (g / g.sum()).view(1, 1, 1, k)
    C = x.shape[1]
    y = F.conv2d(F.pad(x, (k // 2,) * 2 + (0, 0), mode="reflect"),
                 g.expand(C, 1, 1, k), groups=C)
    y = F.conv2d(F.pad(y, (0, 0) + (k // 2,) * 2, mode="reflect"),
                 g.transpose(2, 3).expand(C, 1, k, 1), groups=C)
    return y


def localisation(sal, mask, thr_frac=0.5):
    """IoU and pointing game against a binary lesion mask.

    IoU binarises the saliency at thr_frac of its per-image max.
    Pointing game: does the single most salient pixel land on the lesion?
    Returns (iou, hit) as numpy arrays; entries are NaN where no mask exists.
    """
    B = sal.shape[0]
    s = sal.view(B, -1)
    m = mask.view(B, -1) > 0.5
    thr = s.max(1, keepdim=True).values * thr_frac
    sb = s >= thr
    inter = (sb & m).sum(1).float()
    union = (sb | m).sum(1).float()
    iou = torch.where(union > 0, inter / union,
                      torch.full_like(union, float("nan")))
    peak = s.argmax(1)
    hit = m.gather(1, peak.unsqueeze(1)).squeeze(1).float()
    empty = m.sum(1) == 0
    iou[empty] = float("nan")
    hit[empty] = float("nan")
    return iou.cpu().numpy(), hit.cpu().numpy()
