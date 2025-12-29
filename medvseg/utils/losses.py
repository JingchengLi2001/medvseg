import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # scipy is used for distance transform in boundary loss
    from scipy.ndimage import distance_transform_edt
except Exception as e:  # pragma: no cover
    distance_transform_edt = None
    _scipy_import_error = e


def _targets_to_hard_and_weight(targets: torch.Tensor, eps: float = 1e-6):
    """Convert soft targets (0..1) into:

    - hard labels y in {0,1}
    - per-pixel weights w in (0..1]

    Convention used by this project (vote merge):
      * 0     : background
      * 1     : confident foreground (core)
      * (0,1) : uncertain foreground, where value encodes a lower training weight.

    This fixes the 'too conservative' issue caused by training BCE with soft labels like 0.3.
    """
    if targets.ndim != 4:
        raise ValueError(f"targets must be (B,1,H,W), got {tuple(targets.shape)}")

    # Hard label: any positive value is treated as foreground
    y = (targets > 0).to(dtype=targets.dtype)

    # Weight map: default 1; uncertain pixels take their soft value as weight
    w = torch.ones_like(targets)
    uncertain = (targets > 0) & (targets < 1)
    if uncertain.any():
        w = w.clone()
        w[uncertain] = targets[uncertain].clamp(min=eps, max=1.0)

    return y, w


def dice_loss_weighted(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    """Weighted Dice loss.

    targets may contain uncertain values in (0,1). They are converted to hard labels with weights.
    """
    y, w = _targets_to_hard_and_weight(targets, eps=eps)
    probs = torch.sigmoid(logits)

    # Weighted Dice (soft Dice with weights)
    num = 2.0 * (w * probs * y).sum(dim=(2, 3)) + eps
    den = (w * probs).sum(dim=(2, 3)) + (w * y).sum(dim=(2, 3)) + eps
    return (1.0 - num / den).mean()


def bce_loss_weighted(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    """Weighted BCEWithLogits loss with per-pixel weights."""
    y, w = _targets_to_hard_and_weight(targets, eps=eps)
    per = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    return (per * w).sum() / (w.sum() + eps)


class BoundaryLoss(nn.Module):
    """A simple boundary-aware term based on distance transform.

    Note: This implementation keeps your original behavior but fixes the target binarization
    and supports uncertain-pixel down-weighting.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        if distance_transform_edt is None:
            raise ImportError(
                "scipy is required for BoundaryLoss (distance_transform_edt). "
                "Please install scipy. Original import error: " + str(_scipy_import_error)
            )

        y, w = _targets_to_hard_and_weight(targets)
        probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        b, _, h, w_ = y.shape

        dts = []
        # Compute distance map on CPU per-sample (same as your original version)
        for i in range(b):
            t = (y[i, 0].detach().cpu().numpy() > 0.5)
            dist = distance_transform_edt(t == 0) + distance_transform_edt(t == 1)
            dts.append(torch.from_numpy(dist))

        dt = torch.stack(dts, dim=0).unsqueeze(1).to(logits.device).float()

        # Down-weight uncertain pixels in boundary term as well
        return (probs * dt * w).sum() / (w.sum() + 1e-6)


class ComboLoss(nn.Module):
    """Dice + BCE + Boundary (all support uncertain-pixel weighting).

    - If targets are binary {0,1}, this behaves like a normal combo loss.
    - If targets contain (0,1) values (vote-merge uncertainty), those pixels are treated as
      foreground with lower weight (fixes '胆小/不敢分割').
    """

    def __init__(self, w_dice: float = 1.0, w_bce: float = 1.0, w_boundary: float = 1.0):
        super().__init__()
        self.wd, self.wb, self.wbd = float(w_dice), float(w_bce), float(w_boundary)
        self.bd = BoundaryLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        loss = 0.0
        if self.wd:
            loss = loss + self.wd * dice_loss_weighted(logits, targets)
        if self.wb:
            loss = loss + self.wb * bce_loss_weighted(logits, targets)
        if self.wbd:
            loss = loss + self.wbd * self.bd(logits, targets)
        return loss
