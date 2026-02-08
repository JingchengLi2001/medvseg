"""Auto-tune prediction threshold (PRED_TH) on manual GT frames.

This script is called by endoscopy_oneclick.py. It MUST print the chosen threshold
as the LAST LINE (so the pipeline can parse it). Above that, it can print
diagnostics.

Inputs (positional, for backward compatibility):
  1) dataset_root
  2) split
  3) clip
  4) ckpt
  5) size
  6) eval_only_raw (space-separated list)
  7+) seed_names...

Notes:
- NEG_LIST is read from environment (space-separated list).
- If masks for NEG frames are missing, they are treated as empty GT.
- We report and optimize SEED mean and EVAL mean separately (so you can see both).
"""

from __future__ import annotations

import sys


def _run_with_sysargv(argv: list[str]):
    import os
    from pathlib import Path

    import cv2
    import numpy as np
    import torch

    from medvseg.models.student_unet import StudentUNet

    dataset_root = Path(argv[1])
    split = argv[2]
    clip = argv[3]
    ckpt = argv[4]
    size = int(argv[5])
    eval_only_raw = argv[6].strip()
    seed_names = argv[7:]

    neg_list = os.environ.get("NEG_LIST", "").split()
    eval_only = eval_only_raw.split() if eval_only_raw else []

    # Build group map
    group = {}
    for n in neg_list:
        if n:
            group[n] = "neg"
    for n in eval_only:
        if n:
            group[n] = "eval"
    for n in seed_names:
        if n:
            group[n] = "seed"  # highest priority

    # If no names provided at all, fall back to all frames in frame_dir
    frame_dir = dataset_root / split / clip / "frames"
    mask_dir = dataset_root / split / clip / "masks"

    if not group:
        cand = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            cand.extend([p.name for p in frame_dir.glob(ext)])
        for n in sorted(set(cand)):
            group[n] = "eval"  # treat as eval/general

    names = list(group.keys())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = StudentUNet("resnet34", 3, 1).to(device).eval()
    state = torch.load(ckpt, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    net.load_state_dict(sd)

    probs = []
    gts = []
    groups = []
    used = 0
    skipped_no_frame = 0
    skipped_no_mask = 0

    for name in names:
        im = cv2.imread(str(frame_dir / name), cv2.IMREAD_COLOR)
        if im is None:
            skipped_no_frame += 1
            continue

        gt = cv2.imread(str(mask_dir / name), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            if group.get(name) == "neg":
                gt = np.zeros(im.shape[:2], np.uint8)
            else:
                skipped_no_mask += 1
                continue

        H, W = im.shape[:2]
        x = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = x.to(device)

        with torch.no_grad():
            logit = net(x)
            p = torch.sigmoid(logit)[0, 0].detach().cpu().numpy()

        p_up = cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)
        g = (gt > 0).astype(np.uint8)

        probs.append(p_up)
        gts.append(g)
        groups.append(group.get(name, "eval"))
        used += 1

    print(
        f"[INFO] auto_th samples: requested={len(names)} used={used} "
        f"skip_no_frame={skipped_no_frame} skip_no_mask={skipped_no_mask}"
    )
    if not probs:
        print("")  # keep pipeline behavior predictable
        raise SystemExit

    # Fine grid (threshold is often sensitive near 0.30~0.45)
    ths = np.round(np.arange(0.15, 0.651, 0.01), 2)

    def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
        a = float(pred.sum())
        b = float(gt.sum())
        if a == 0.0 and b == 0.0:
            return 1.0
        if b == 0.0 and a > 0.0:
            return 0.0
        inter = float((pred & gt).sum())
        return (2.0 * inter) / (a + b + 1e-6)

    best = {
        "th": 0.5,
        "obj": -1.0,
        "seed_mean": 0.0,
        "eval_mean": 0.0,
        "neg_mean": 0.0,
        "all_mean": 0.0,
    }

    have_seed = any(g == "seed" for g in groups)
    have_eval = any(g == "eval" for g in groups)
    have_neg = any(g == "neg" for g in groups)

    for th in ths:
        seed_ds = []
        eval_ds = []
        neg_ds = []
        all_ds = []

        for p, g, grp in zip(probs, gts, groups):
            pred = (p >= th).astype(np.uint8)
            d = _dice(pred, g)
            all_ds.append(d)
            if grp == "seed":
                seed_ds.append(d)
            elif grp == "eval":
                eval_ds.append(d)
            else:
                neg_ds.append(d)

        seed_mean = float(np.mean(seed_ds)) if seed_ds else float("nan")
        eval_mean = float(np.mean(eval_ds)) if eval_ds else float("nan")
        neg_mean = float(np.mean(neg_ds)) if neg_ds else float("nan")
        all_mean = float(np.mean(all_ds)) if all_ds else 0.0

        # Objective: keep BOTH seed and eval high; lightly enforce NEG emptiness.
        # If one group doesn't exist, fall back to the other.
        if have_seed and have_eval:
            # harmonic mean for stability (penalize the lower one)
            h = (2.0 * seed_mean * eval_mean) / (seed_mean + eval_mean + 1e-9)
            obj = 0.9 * h
        elif have_seed:
            obj = seed_mean
        else:
            obj = eval_mean

        if have_neg:
            # If NEG predicts empty => dice=1; if FP exists => dice drops to 0.
            obj = 0.9 * obj + 0.1 * neg_mean

        if obj > best["obj"]:
            best = {
                "th": float(th),
                "obj": float(obj),
                "seed_mean": float(seed_mean) if seed_ds else float("nan"),
                "eval_mean": float(eval_mean) if eval_ds else float("nan"),
                "neg_mean": float(neg_mean) if neg_ds else float("nan"),
                "all_mean": float(all_mean),
            }

    print(
        "[AUTO_TH] "
        f"best_th={best['th']:.2f} obj={best['obj']:.4f} "
        f"seed_mean={best['seed_mean']:.4f} eval_mean={best['eval_mean']:.4f} "
        f"neg_mean={best['neg_mean']:.4f} all_mean={best['all_mean']:.4f}"
    )

    # IMPORTANT: the pipeline parses the LAST LINE as float threshold
    print(f"{best['th']:.4f}")


def main():
    # keep historical invocation style
    _run_with_sysargv(sys.argv)


if __name__ == "__main__":
    main()
