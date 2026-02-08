from __future__ import annotations

import argparse
import contextlib
import copy
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from medvseg.data.datasets import FrameMaskDataset
from medvseg.models.student_unet import StudentUNet
from medvseg.utils.common import ensure_dir
from medvseg.utils.losses import ComboLoss
from medvseg.utils.metrics import dice_score, iou_score


def _autocast_ctx(device: str):
    """Compatibility wrapper for AMP autocast."""
    if device == "cuda":
        return torch.autocast(device_type="cuda", enabled=True)
    return contextlib.nullcontext()


def _grad_scaler(device: str):
    """Compatibility wrapper for AMP GradScaler."""
    if device == "cuda":
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=True)

    class _Dummy:
        def scale(self, x):
            return x

        def step(self, opt):
            opt.step()

        def update(self):
            return None

    return _Dummy()


@torch.no_grad()
def _ema_update(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float):
    msd = model.state_dict()
    esd = ema_model.state_dict()
    for k, v in esd.items():
        if k not in msd:
            continue
        src = msd[k]
        if not torch.is_floating_point(v):
            v.copy_(src)
        else:
            v.mul_(decay).add_(src, alpha=(1.0 - decay))


def train(
    data_root: str,
    val_root: str,
    save_dir: str,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(save_dir)

    # Optional speed knobs (safe defaults)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    train_ds = FrameMaskDataset(data_root, transform="train")
    val_ds = FrameMaskDataset(val_root, transform="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = StudentUNet("resnet34", 3, 1).to(device)

    # Loss: soft-target Dice+BCE with uncertainty down-weighting.
    # Loss knobs (backward compatible):
    #   BG_WEIGHT or BCE_BG_W: penalize false positives (useful if you over-seg).
    #   UNC_WEIGHT: downweight uncertain soft labels.
    #   BOUNDARY_W: boundary loss weight (0 keeps old behavior).
    bg_weight = float(os.environ.get("BG_WEIGHT", os.environ.get("BCE_BG_W", "1.2")))
    unc_weight = float(os.environ.get("UNC_WEIGHT", os.environ.get("UNC_W", "0.5")))
    boundary_w = float(os.environ.get("BOUNDARY_W", "0.0"))
    w_bce = float(os.environ.get("W_BCE", "1.0"))
    w_dice = float(os.environ.get("W_DICE", "1.0"))
    loss_fn = ComboLoss(w_bce, w_dice, boundary_w, unc_weight=unc_weight, bg_weight=bg_weight)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = _grad_scaler(device)

    # EMA (usually improves generalization with almost no downside)
    use_ema = os.environ.get("TRAIN_EMA", "1").strip() in ("1", "true", "True")
    ema_decay = float(os.environ.get("TRAIN_EMA_DECAY", "0.999"))
    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    # LR scheduler (safe default: reduce on plateau)
    sched_mode = os.environ.get("LR_SCHED", "plateau").strip().lower()
    scheduler = None
    if sched_mode == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=3, threshold=1e-4, min_lr=1e-6
        )
    elif sched_mode == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    # else: no scheduler

    best = -1.0
    best_path = Path(save_dir) / "best.ckpt"

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, ncols=110, desc=f"Epoch {ep}/{epochs}")

        for x, y, _ in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)

            with _autocast_ctx(device):
                pred = model(x)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()

            # Optional grad clipping
            clip_g = float(os.environ.get("CLIP_GRAD", "0"))
            if clip_g and clip_g > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_g)

            scaler.step(opt)
            scaler.update()

            if ema_model is not None:
                _ema_update(ema_model, model, decay=ema_decay)

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")

        # Validation (use EMA weights if enabled)
        eval_model = ema_model if ema_model is not None else model
        eval_model.eval()

        dices, ious = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                pred = eval_model(x)

                # For metrics, treat soft pseudo-label >=0.5 as foreground.
                y_metric = (y >= 0.5).float()
                dices.append(dice_score(pred, y_metric))
                ious.append(iou_score(pred, y_metric))

        md, mi = sum(dices) / len(dices), sum(ious) / len(ious)
        print(f"Val Dice={md:.4f} IoU={mi:.4f}")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(md)
            else:
                scheduler.step()

        if md > best:
            best = md
            # Save EMA weights if available (usually better at inference)
            state = (ema_model.state_dict() if ema_model is not None else model.state_dict())
            torch.save({"model": state}, best_path)
            print("** Saved best")

    print(f"[DONE] Best Val Dice={best:.4f} -> {best_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()
    train(**vars(args))
