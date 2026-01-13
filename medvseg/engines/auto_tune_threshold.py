"""Auto-tune prediction threshold (PRED_TH) on MANUAL frames.

Prints best threshold as the LAST stdout line (pipeline parses it).
Any debug goes to stderr.
"""

from __future__ import annotations

import argparse


def _run_with_sysargv(argv: list[str]):
    import sys
    sys.argv = argv

    import os
    from pathlib import Path

    import cv2
    import numpy as np
    import torch

    from medvseg.models.student_unet import StudentUNet

    dataset_root = Path(sys.argv[1])
    split = sys.argv[2]
    clip = sys.argv[3]
    ckpt = sys.argv[4]
    size = int(sys.argv[5])
    eval_only_raw = sys.argv[6].strip()
    seed_names = sys.argv[7:]

    # optional NEG constraints
    neg_fp_max_limit = os.environ.get("NEG_FP_MAX", "").strip()
    neg_fp_max_limit = float(neg_fp_max_limit) if neg_fp_max_limit else None

    neg_list = os.environ.get("NEG_LIST", "").split()
    eval_only = eval_only_raw.split() if eval_only_raw else []

    # build unique name list
    names = []
    seen = set()
    for n in (seed_names + eval_only + neg_list):
        if not n or n in seen:
            continue
        seen.add(n)
        names.append(n)

    frame_dir = dataset_root / split / clip / "frames"
    mask_dir = dataset_root / split / clip / "masks"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = StudentUNet("resnet34", 3, 1).to(device).eval()
    state = torch.load(ckpt, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    net.load_state_dict(sd)

    probs = []
    gts = []
    is_neg_flags = []

    for name in names:
        im = cv2.imread(str(frame_dir / name), cv2.IMREAD_COLOR)
        if im is None:
            continue

        gt = cv2.imread(str(mask_dir / name), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            if name in neg_list:
                gt = np.zeros(im.shape[:2], np.uint8)
            else:
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

        probs.append(p_up.astype(np.float32))
        gts.append(g)
        is_neg_flags.append(int(g.sum() == 0))

    if not probs:
        # keep pipeline behavior: output empty then exit
        print("")
        raise SystemExit

    # candidate thresholds (稳健范围，避免极端)
    ths = np.linspace(0.2, 0.8, 13)

    best_th = 0.5
    best_mean_dice = -1.0
    best_pos_dice = -1.0
    best_neg_fp_max = 1e9
    any_feasible = False

    # precompute which frames are NEG by GT (empty GT)
    neg_indices = [i for i, f in enumerate(is_neg_flags) if f == 1]
    pos_indices = [i for i, f in enumerate(is_neg_flags) if f == 0]

    for th in ths:
        dices = []
        pos_dices = []
        neg_fps = []

        for p, g in zip(probs, gts):
            pred = (p >= th).astype(np.uint8)
            inter = int((pred & g).sum())
            a = int(pred.sum())
            b = int(g.sum())

            if b == 0:
                # NEG frame: dice=1 if pred empty else 0
                d = 1.0 if a == 0 else 0.0
                neg_fps.append(a / float(g.size))
            else:
                d = (2.0 * inter) / (a + b + 1e-6)
                pos_dices.append(d)

            dices.append(d)

        mean_dice = float(np.mean(dices)) if dices else 0.0
        pos_dice = float(np.mean(pos_dices)) if pos_dices else 0.0
        neg_fp_max = float(np.max(neg_fps)) if neg_fps else 0.0

        feasible = True
        if (neg_fp_max_limit is not None) and neg_fps:
            feasible = (neg_fp_max <= neg_fp_max_limit)

        if feasible:
            any_feasible = True
            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                best_pos_dice = pos_dice
                best_neg_fp_max = neg_fp_max
                best_th = float(th)
        else:
            # if no feasible threshold exists, fall back to minimizing neg_fp_max (then maximize mean_dice)
            if not any_feasible:
                if (neg_fp_max < best_neg_fp_max) or (neg_fp_max == best_neg_fp_max and mean_dice > best_mean_dice):
                    best_mean_dice = mean_dice
                    best_pos_dice = pos_dice
                    best_neg_fp_max = neg_fp_max
                    best_th = float(th)

    # stats for reporting
    neg_fg_pct = 0.0
    if neg_indices:
        cnt_nonempty = 0
        for i in neg_indices:
            pred = (probs[i] >= best_th).astype(np.uint8)
            if int(pred.sum()) > 0:
                cnt_nonempty += 1
        neg_fg_pct = cnt_nonempty / float(len(neg_indices))

    # debug to stderr (safe)
    msg = (
        f"[AUTO_TH] best_th={best_th:.4f} mean_dice={best_mean_dice:.4f} "
        f"pos_dice={best_pos_dice:.4f} neg_fp_max={best_neg_fp_max:.6f} "
        f"neg_fg_pct={neg_fg_pct:.3f}"
    )
    if neg_fp_max_limit is not None and neg_indices:
        msg += f" (NEG_FP_MAX={neg_fp_max_limit:.6f}, feasible={any_feasible})"
        if not any_feasible:
            msg += " [WARN no feasible th, using min neg_fp_max]"
    print(msg, file=sys.stderr)

    # IMPORTANT: keep numeric threshold as LAST stdout line
    print(f"{best_th:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root")
    ap.add_argument("split")
    ap.add_argument("clip")
    ap.add_argument("ckpt")
    ap.add_argument("size", type=int)
    ap.add_argument("eval_only_list", nargs="?", default="")
    ap.add_argument("seed_names", nargs="*", default=[])
    args = ap.parse_args()

    argv = [
        "auto_tune_threshold.py",
        args.dataset_root,
        args.split,
        args.clip,
        args.ckpt,
        str(args.size),
        str(args.eval_only_list),
    ] + list(args.seed_names)
    _run_with_sysargv(argv)


if __name__ == "__main__":
    main()
