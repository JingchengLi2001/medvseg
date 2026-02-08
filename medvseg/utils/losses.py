from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # scipy is used for distance transform in boundary loss
    from scipy.ndimage import distance_transform_edt
except Exception as e:  # pragma: no cover
    distance_transform_edt = None
    _scipy_import_error = e


def _weight_map(
    targets: torch.Tensor,
    unc_weight: float = 0.5,
    bg_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return a per-pixel weight map for targets in [0,1].

    Conventions used by this project (vote merge):
      * 0     : background
      * 1     : confident foreground (core)
      * (0,1) : uncertain pseudo-label probability (soft)
    We down-weight uncertain pixels to reduce pseudo-label noise,
    and optionally up-weight strict background pixels to penalize false positives.
    """
    if targets.ndim != 4:
        raise ValueError(f"targets must be (B,1,H,W), got {tuple(targets.shape)}")

    t = targets.clamp(0.0, 1.0)
    w = torch.ones_like(t)

    # uncertain pixels: 0 < t < 1
    if unc_weight < 1.0:
        unc = (t > 0.0) & (t < 1.0)
        if unc.any():
            w = w.clone()
            w[unc] = float(unc_weight)

    # strict background pixels: t == 0
    if bg_weight != 1.0:
        bg = (t <= eps)
        if bg.any():
            w = w.clone()
            w[bg] = w[bg] * float(bg_weight)

    return w


def dice_loss_soft(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6):
    """Soft Dice loss that supports soft targets in [0,1]."""
    t = targets.clamp(0.0, 1.0)
    p = torch.sigmoid(logits)

    if weight is None:
        num = 2.0 * (p * t).sum(dim=(2, 3)) + eps
        den = (p.sum(dim=(2, 3)) + t.sum(dim=(2, 3)) + eps)
    else:
        w = weight
        num = 2.0 * (w * p * t).sum(dim=(2, 3)) + eps
        den = (w * p).sum(dim=(2, 3)) + (w * t).sum(dim=(2, 3)) + eps

    return (1.0 - num / den).mean()


def bce_loss_soft(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6):
    """BCEWithLogits loss that supports soft targets in [0,1] + optional per-pixel weights."""
    t = targets.clamp(0.0, 1.0)
    per = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    if weight is None:
        return per.mean()
    w = weight
    return (per * w).sum() / (w.sum() + eps)


class BoundaryLoss(nn.Module):
    """Boundary-aware term based on distance transform (optional, requires scipy)."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        if distance_transform_edt is None:
            raise ImportError(
                "scipy is required for BoundaryLoss (distance_transform_edt). "
                "Please install scipy. Original import error: " + str(_scipy_import_error)
            )

        t = targets.clamp(0.0, 1.0)
        # boundary term uses hard mask (core) to avoid being dominated by soft uncertainty
        hard = (t >= 0.5).to(dtype=t.dtype)

        probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        b, _, _, _ = hard.shape

        dts = []
        # Compute distance map on CPU per-sample (same as original)
        for i in range(b):
            m = (hard[i, 0].detach().cpu().numpy() > 0.5)
            dist = distance_transform_edt(m == 0) + distance_transform_edt(m == 1)
            dts.append(torch.from_numpy(dist))

        dt = torch.stack(dts, dim=0).unsqueeze(1).to(logits.device).float()
        return (probs * dt).mean()


class ComboLoss(nn.Module):
    """Dice + BCE (+ optional Boundary) with robust handling of soft pseudo-labels.

    This is designed for your vote-merge output, where mask values may be in (0,1).
    Using soft targets avoids the over-segmentation issue (FP-heavy) that happens when
    all uncertain pixels are forced to be foreground.
    """

    def __init__(
        self,
        w_dice: float = 1.0,
        w_bce: float = 1.0,
        w_boundary: float = 0.0,
        *,
        unc_weight: float = 0.5,
        bg_weight: float = 1.0,
    ):
        super().__init__()
        self.wd, self.wb, self.wbd = float(w_dice), float(w_bce), float(w_boundary)
        self.unc_weight = float(unc_weight)
        self.bg_weight = float(bg_weight)
        self.bd = BoundaryLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        w = _weight_map(targets, unc_weight=self.unc_weight, bg_weight=self.bg_weight)
        loss = 0.0
        if self.wd:
            loss = loss + self.wd * dice_loss_soft(logits, targets, weight=w)
        if self.wb:
            loss = loss + self.wb * bce_loss_soft(logits, targets, weight=w)
        if self.wbd:
            loss = loss + self.wbd * self.bd(logits, targets)
        return loss
