"""Auto-tune prediction threshold (PRED_TH) on MANUAL frames.

Extracted from run.sh. Prints best threshold."""

from __future__ import annotations

import argparse


def _run_with_sysargv(argv: list[str]):
    import sys
    sys.argv = argv
    import os, sys, cv2, numpy as np, torch
    from pathlib import Path
    from medvseg.models.student_unet import StudentUNet

    dataset_root = Path(sys.argv[1])
    split = sys.argv[2]
    clip = sys.argv[3]
    ckpt = sys.argv[4]
    size = int(sys.argv[5])
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
    mask_dir = dataset_root/split/clip/'masks'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = StudentUNet('resnet34', 3, 1).to(device).eval()
    state = torch.load(ckpt, map_location=device)
    sd = state['model'] if isinstance(state, dict) and 'model' in state else state
    net.load_state_dict(sd)

    probs = []
    gts = []
    for name in names:
        im = cv2.imread(str(frame_dir/name), cv2.IMREAD_COLOR)
        if im is None:
            continue
        gt = cv2.imread(str(mask_dir/name), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            if name in neg_list:
                gt = np.zeros(im.shape[:2], np.uint8)
            else:
                continue

        H,W = im.shape[:2]
        x = cv2.resize(im, (size,size), interpolation=cv2.INTER_LINEAR)[:,:,::-1].copy()
        x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).float()/255.0
        x = x.to(device)
        with torch.no_grad():
            logit = net(x)
            p = torch.sigmoid(logit)[0,0].detach().cpu().numpy()

        p_up = cv2.resize(p, (W,H), interpolation=cv2.INTER_LINEAR)
        g = (gt>0).astype(np.uint8)
        probs.append(p_up)
        gts.append(g)

    if not probs:
        print('')
        raise SystemExit

    ths = np.linspace(0.2, 0.8, 13)

    best_th = 0.5
    best_d = -1.0

    for th in ths:
        ds = []
        for p,g in zip(probs,gts):
            pred = (p>=th).astype(np.uint8)
            inter = (pred & g).sum()
            a = pred.sum(); b = g.sum()
            # handle empty-GT: if both empty => dice=1, else normal formula
            if b == 0 and a == 0:
                d = 1.0
            elif b == 0 and a > 0:
                d = 0.0
            else:
                d = (2*inter)/(a+b+1e-6)
            ds.append(d)
        md = float(np.mean(ds))
        if md > best_d:
            best_d = md
            best_th = float(th)

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

    argv = ["auto_tune_threshold.py",
            args.dataset_root, args.split, args.clip, args.ckpt, str(args.size),
            str(args.eval_only_list)] + list(args.seed_names)
    _run_with_sysargv(argv)

if __name__ == "__main__":
    main()

