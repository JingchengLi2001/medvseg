"""Evaluate model on MANUAL seed frames only and print per-frame dice/IoU.

Extracted from run.sh."""

from __future__ import annotations

import argparse


def _run_with_sysargv(argv: list[str]):
    import sys
    sys.argv = argv
    import sys, os, cv2, numpy as np, torch
    from pathlib import Path
    from medvseg.models.student_unet import StudentUNet

    frame_dir = Path(sys.argv[1]); mask_dir = Path(sys.argv[2])
    ckpt = sys.argv[3]; size=int(sys.argv[4]); th=float(sys.argv[5])
    seed_names = sys.argv[6:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = StudentUNet("resnet34", 3, 1).to(device).eval()
    state = torch.load(ckpt, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    net.load_state_dict(sd)

    def dice_iou(pred, gt):
        pred = (pred>0).astype(np.uint8)
        gt = (gt>0).astype(np.uint8)
        inter = (pred & gt).sum()
        a = pred.sum(); b = gt.sum()
        dice = (2*inter) / (a+b+1e-6)
        iou = inter / (a+b-inter+1e-6)
        return float(dice), float(iou), int(a), int(b)

    ds=[]; is_=[]
    for name in seed_names:
        fp = frame_dir/name
        mp = mask_dir/name
        im = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if im is None or gt is None:
            continue
        H,W = im.shape[:2]
        x = cv2.resize(im, (size,size), interpolation=cv2.INTER_LINEAR)[:,:,::-1].copy()
        x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).float()/255.0
        x = x.to(device)
        with torch.no_grad():
            logits = net(x)
            prob = torch.sigmoid(logits)[0,0].detach().cpu().numpy()
        pred = (prob >= th).astype(np.uint8)*255
        pred_up = cv2.resize(pred, (W,H), interpolation=cv2.INTER_NEAREST)
        d,i,a,b = dice_iou(pred_up, gt)
        ds.append(d); is_.append(i)
        print(f"[SEED] {name} dice={d:.4f} iou={i:.4f} pred_fg={a} gt_fg={b}")

    if ds:
        print(f"[SEED-MEAN] dice={np.mean(ds):.4f} iou={np.mean(is_):.4f} (th={th})")
    else:
        print("[WARN] no seed metrics computed.")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame_dir")
    ap.add_argument("mask_dir")
    ap.add_argument("ckpt")
    ap.add_argument("size", type=int)
    ap.add_argument("th", type=float)
    ap.add_argument("seed_names", nargs="*", default=[])
    args = ap.parse_args()

    argv = ["eval_manual_seeds.py",
            args.frame_dir, args.mask_dir, args.ckpt, str(args.size), str(args.th)] + list(args.seed_names)
    _run_with_sysargv(argv)

if __name__ == "__main__":
    main()

