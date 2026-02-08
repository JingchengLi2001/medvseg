"""Evaluate model on manually-labeled frames and print Dice/IoU.

This is used by endoscopy_oneclick.py to report:
- seed mean Dice (SEED_LIST)
- eval-list mean Dice (EVAL_ONLY_LIST)

CLI (backward compatible with old positional call):
  python -m medvseg.engines.eval_manual_seeds FRAMES_DIR MASKS_DIR CKPT SIZE TH name1 name2 ...

Extra flags:
  --label {SEED|EVAL|...}  changes the print prefix
  --min-area N             remove tiny connected components after thresholding (0 disables)
"""

from __future__ import annotations

import argparse


def _dice_iou(pred: "np.ndarray", gt: "np.ndarray") -> tuple[float, float]:
    import numpy as np
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return (1.0, 1.0)
    dice = (2 * inter) / (pred.sum() + gt.sum() + 1e-6)
    iou = inter / (union + 1e-6)
    return float(dice), float(iou)


def _rm_small_cc(mask: "np.ndarray", min_area: int) -> "np.ndarray":
    import cv2
    import numpy as np
    if min_area <= 0:
        return mask
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"), connectivity=8)
    if num <= 1:
        return mask
    out = np.zeros_like(mask, dtype="uint8")
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[lab == i] = 1
    return out


def main():
    import os
    from pathlib import Path

    import cv2
    import numpy as np
    import torch

    from medvseg.models.student_unet import StudentUNet

    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("masks_dir")
    ap.add_argument("ckpt")
    ap.add_argument("size", type=int)
    ap.add_argument("th", type=float)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--label", default="SEED")
    ap.add_argument("--min-area", type=int, default=int(os.environ.get("EVAL_MIN_AREA", "0")))
    args = ap.parse_args()

    frame_dir = Path(args.frames_dir)
    mask_dir = Path(args.masks_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = StudentUNet("resnet34", 3, 1).to(device).eval()
    state = torch.load(args.ckpt, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    net.load_state_dict(sd)

    ds, is_ = [], []
    for name in args.names:
        fp = frame_dir / name
        mp = mask_dir / name
        im = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if im is None or gt is None:
            print(f"[{args.label}] {name} missing frame/mask, skip")
            continue

        H, W = im.shape[:2]
        x = cv2.resize(im, (args.size, args.size), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = x.to(device)

        with torch.no_grad():
            p = torch.sigmoid(net(x))[0, 0].detach().cpu().numpy()

        p = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)
        pred = (p >= args.th).astype("uint8")
        pred = _rm_small_cc(pred, args.min_area)

        gt = (gt > 0).astype("uint8")
        d, i = _dice_iou(pred, gt)
        ds.append(d); is_.append(i)
        print(f"[{args.label}] {name} dice={d:.4f} iou={i:.4f} pred_fg={int(pred.sum())} gt_fg={int(gt.sum())}")

    if ds:
        print(f"[{args.label}-MEAN] dice={float(np.mean(ds)):.4f} iou={float(np.mean(is_)):.4f} (th={args.th:.2f})")


if __name__ == "__main__":
    main()
