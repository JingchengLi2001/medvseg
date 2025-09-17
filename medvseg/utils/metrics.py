import torch

def _th(x, thr=0.5):
    return (torch.sigmoid(x) >= thr).float()

def dice_score(logits, targets, thr=0.5, eps=1e-6):
    p = _th(logits, thr)
    num = 2 * (p * targets).sum(dim=(2, 3)) + eps
    den = (p.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + eps)
    return (num / den).mean().item()

def iou_score(logits, targets, thr=0.5, eps=1e-6):
    p = _th(logits, thr)
    inter = (p * targets).sum(dim=(2, 3))
    union = (p + targets - p * targets).sum(dim=(2, 3)) + eps
    return (inter / union).mean().item()
