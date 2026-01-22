"""Summarize motion gate behavior from overlay_video exported artifacts.

Inputs:
  - a CSV produced by overlay_video.py via --save-metrics
  - (optional) a directory of per-frame binary masks produced by --save-masks

It prints:
  - break rate, disabled rate
  - non-zero mask ratios overall / on break frames / on non-break frames
  - basic metric stats (flow_p95, conf_mean, blur_var) split by break vs non-break
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def _safe_float(x: str):
    try:
        return float(x)
    except Exception:
        return None


def _read_rows(csv_path: Path):
    import csv
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _mask_nonzero(mask_path: Path) -> int:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return 0
    return int((m > 0).any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_csv", type=str, help="CSV path saved by overlay_video --save-metrics")
    ap.add_argument("--masks-dir", type=str, default="", help="dir saved by overlay_video --save-masks")
    args = ap.parse_args()

    csv_path = Path(args.metrics_csv)
    if not csv_path.exists():
        raise SystemExit(f"[ERR] metrics csv not found: {csv_path}")

    rows = _read_rows(csv_path)
    if not rows:
        print("[WARN] empty metrics csv:", str(csv_path))
        return

    masks_dir = Path(args.masks_dir) if args.masks_dir else None
    has_masks = masks_dir is not None and masks_dir.exists()

    n = len(rows)
    breaks = 0
    disabled = 0
    nonzero_all = 0
    nonzero_break = 0
    nonzero_stable = 0
    n_break = 0
    n_stable = 0

    # metric aggregates
    def agg_init():
        return {"flow_p95": [], "conf_mean": [], "conf_pct": [], "blur_var": []}

    agg_b = agg_init()
    agg_s = agg_init()

    for r in rows:
        br = int(float(r.get("break", "0") or 0) > 0.5)
        en = int(float(r.get("enabled", "1") or 1) > 0.5)
        breaks += br
        disabled += (1 - en)

        if br:
            n_break += 1
        else:
            n_stable += 1

        if has_masks:
            idx = int(float(r.get("frame_idx", "0") or 0))
            mp = masks_dir / f"{idx:05d}.png"
            nz = _mask_nonzero(mp)
            nonzero_all += nz
            if br:
                nonzero_break += nz
            else:
                nonzero_stable += nz

        # metrics
        for k in agg_b.keys():
            v = _safe_float(r.get(k, ""))
            if v is None:
                continue
            if br:
                agg_b[k].append(v)
            else:
                agg_s[k].append(v)

    def pct(x, d):
        return 0.0 if d <= 0 else 100.0 * float(x) / float(d)

    print(f"[GATE] frames={n} break={breaks} ({pct(breaks,n):.1f}%) disabled={disabled} ({pct(disabled,n):.1f}%)")

    if has_masks:
        print(f"[MASK] nonzero(all)={nonzero_all}/{n} ({pct(nonzero_all,n):.2f}%)")
        if n_break > 0:
            print(f"[MASK] nonzero(break)={nonzero_break}/{n_break} ({pct(nonzero_break,n_break):.2f}%)")
        if n_stable > 0:
            print(f"[MASK] nonzero(stable)={nonzero_stable}/{n_stable} ({pct(nonzero_stable,n_stable):.2f}%)")

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    # show a tiny table of split means
    keys = ["flow_p95", "conf_mean", "conf_pct", "blur_var"]
    parts = [("stable", agg_s), ("break", agg_b)]
    print("[METRIC] mean values (stable vs break):")
    for k in keys:
        ms = mean(agg_s[k])
        mb = mean(agg_b[k])
        if ms is None and mb is None:
            continue
        ms_s = "NA" if ms is None else f"{ms:.4f}"
        mb_s = "NA" if mb is None else f"{mb:.4f}"
        print(f" - {k}: stable={ms_s} break={mb_s}")


if __name__ == "__main__":
    main()
