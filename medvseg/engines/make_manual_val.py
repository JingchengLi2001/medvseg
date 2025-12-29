"""Build a small MANUAL validation set from original labeled frames.

Extracted from run.sh so it's reusable.
Copies frames+GT masks from {DATASET_ROOT}/{split}/{clip} into outputs/val_manual/{split}/{clip}.
Supports NEG_LIST and EVAL_ONLY_LIST."""

from __future__ import annotations

import argparse


def _run_with_sysargv(argv: list[str]):
    import sys
    sys.argv = argv
    import os, sys, cv2
    from pathlib import Path
    import numpy as np

    dataset_root = Path(sys.argv[1])
    split = sys.argv[2]
    clip = sys.argv[3]
    size = int(sys.argv[4])
    val_root = Path(sys.argv[5])
    eval_only_raw = sys.argv[6].strip()
    seed_names = sys.argv[7:]

    neg_list = os.environ.get('NEG_LIST', '').split()
    eval_only = eval_only_raw.split() if eval_only_raw else []

    names = []
    seen = set()
    for n in (seed_names + eval_only + neg_list):
        if not n or n in seen:
            continue
        seen.add(n)
        names.append(n)

    frame_dir = dataset_root/split/clip/'frames'
    mask_dir  = dataset_root/split/clip/'masks'

    dst_clip = val_root/split/clip
    dst_f = dst_clip/'frames'
    dst_m = dst_clip/'masks'
    dst_f.mkdir(parents=True, exist_ok=True)
    dst_m.mkdir(parents=True, exist_ok=True)

    kept = 0
    for name in names:
        fp = frame_dir/name
        im = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if im is None:
            continue
        H,W = im.shape[:2]
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)

        mp = mask_dir/name
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)

        if m is None:
            # If it's a NEG frame, create an empty mask. Otherwise skip.
            if name in neg_list:
                m = np.zeros((H, W), np.uint8)
            else:
                continue

        m = (m > 0).astype('uint8') * 255
        m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(dst_f/name), im)
        cv2.imwrite(str(dst_m/name), m)
        kept += 1

    print(f"[OK] manual val set: {dst_clip} kept={kept} (includes NEG={len(neg_list)})")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root")
    ap.add_argument("split")
    ap.add_argument("clip")
    ap.add_argument("size", type=int)
    ap.add_argument("val_root")
    ap.add_argument("eval_only_list", nargs="?", default="")
    ap.add_argument("seed_names", nargs="*", default=[])
    args = ap.parse_args()

    argv = ["make_manual_val.py",
            args.dataset_root, args.split, args.clip, str(args.size),
            args.val_root, str(args.eval_only_list)] + list(args.seed_names)
    _run_with_sysargv(argv)

if __name__ == "__main__":
    main()

