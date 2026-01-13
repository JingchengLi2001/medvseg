"""Merge multi-seed pseudo labels into a single training set.

This file replaces the large heredoc inside run.sh / pipeline.
It supports three strategies (env MERGE_STRATEGY): union / nearest / vote.
It also supports forcing some frames as NEG via env NEG_LIST.
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

    out_multi = Path(sys.argv[1])
    split = sys.argv[2]
    clip = sys.argv[3]
    size = int(sys.argv[4])
    frame_dir = Path(sys.argv[5])
    minpix = int(sys.argv[6])
    roots = [Path(p) for p in sys.argv[7:]]

    strategy = os.environ.get("MERGE_STRATEGY", "nearest")
    vote_min = int(os.environ.get("VOTE_MIN", "0") or "0")
    vote_min_seeds = int(os.environ.get("VOTE_MIN_SEEDS", "1") or "1")
    merge_topk = int(os.environ.get("MERGE_TOPK", "0") or "0")
    uncert_val = float(os.environ.get("UNCERT_VAL", "0.30") or "0.30")

    eval_only = set(os.environ.get("EVAL_ONLY_LIST", "").split())
    neg_list = set(os.environ.get("NEG_LIST", "").split())

    seed_names = (os.environ.get("MERGE_SEEDS", "").split() or os.environ.get("SEED_LIST", "").split())

    if not roots:
        raise SystemExit("[ERR] No roots for merge.")

    dst_clip = out_multi / split / clip
    (dst_clip / "frames").mkdir(parents=True, exist_ok=True)
    (dst_clip / "masks").mkdir(parents=True, exist_ok=True)

    dst_f = dst_clip / "frames"
    dst_m = dst_clip / "masks"

    # ---------- helpers ----------
    def _frame_index(name: str, name_to_idx: dict[str, int]):
        return name_to_idx.get(name, 10**9)

    def _read_mask(path: Path):
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        if (m.shape[0] != size) or (m.shape[1] != size):
            m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
        return (m > 0).astype(np.uint8)

    def _read_conf(path: Path):
        c = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if c is None:
            return None
        if (c.shape[0] != size) or (c.shape[1] != size):
            c = cv2.resize(c, (size, size), interpolation=cv2.INTER_LINEAR)
        c = c.astype(np.float32) / 255.0
        return np.clip(c, 0.0, 1.0)

    def _write_pair(name: str, merged_u8: np.ndarray):
        # drop tiny positives to keep train set clean
        if int((merged_u8 > 0).sum()) < minpix:
            return "small"

        cv2.imwrite(str(dst_m / name), merged_u8.astype(np.uint8))

        fp = frame_dir / name
        if not fp.exists():
            return "missing"
        im = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if im is None:
            return "missing"
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(dst_f / name), im)
        return "ok"

    def _write_neg_pair(name: str):
        # Force NEG: keep frame, write empty mask, NEVER drop by minpix
        fp = frame_dir / name
        if not fp.exists():
            return "missing"
        im = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if im is None:
            return "missing"
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(dst_f / name), im)

        empty = np.zeros((size, size), np.uint8)
        cv2.imwrite(str(dst_m / name), empty)
        return "ok"

    # collect all mask names across roots
    names = set()
    for r in roots:
        mdir = r / split / clip / "masks"
        if not mdir.exists():
            continue
        for p in mdir.glob("*.png"):
            names.add(p.name)

    # ensure NEG frames are included even if no mask exists in roots
    for n in neg_list:
        if n:
            names.add(n)

    names = sorted(names)
    if not names:
        raise SystemExit(f"[ERR] No masks to merge. Check filter outputs under: {roots}")

    name_to_idx = {n: i for i, n in enumerate(names)}

    seed_indices = []
    if seed_names:
        if len(seed_names) != len(roots):
            print(f"[WARN] SEED_LIST ({len(seed_names)}) != roots ({len(roots)}); using min length.")
        for s in seed_names:
            seed_indices.append(_frame_index(s, name_to_idx))

    kept = 0
    forced_neg = 0
    skipped_eval = 0
    skipped_small = 0
    skipped_missing = 0
    used_conf_frames = 0

    for name in names:
        if name in eval_only:
            skipped_eval += 1
            continue

        # Force NEG first (do not participate in merge strategy)
        if name in neg_list:
            status = _write_neg_pair(name)
            if status == "ok":
                kept += 1
                forced_neg += 1
            else:
                skipped_missing += 1
            continue

        # strategy: nearest
        if strategy == "nearest":
            frame_idx = _frame_index(name, name_to_idx)
            best = None
            best_dist = None
            for i, r in enumerate(roots):
                mp = r / split / clip / "masks" / name
                if not mp.exists():
                    continue
                dist = abs(frame_idx - seed_indices[i]) if i < len(seed_indices) else 0
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best = mp
            if best is None:
                skipped_missing += 1
                continue
            m = _read_mask(best)
            if m is None:
                skipped_missing += 1
                continue
            merged_u8 = (m * 255).astype(np.uint8)
            status = _write_pair(name, merged_u8)
            if status == "ok":
                kept += 1
            elif status == "small":
                skipped_small += 1
            else:
                skipped_missing += 1
            continue

        # strategy: vote (confidence-weighted)
        if strategy == "vote":
            roots_use = roots
            if merge_topk > 0 and seed_indices:
                frame_idx = _frame_index(name, name_to_idx)
                dists = []
                for i, r in enumerate(roots):
                    if i >= len(seed_indices):
                        continue
                    dists.append((abs(frame_idx - seed_indices[i]), r))
                dists.sort(key=lambda x: x[0])
                roots_use = [r for _, r in dists[:merge_topk]]

            count = None
            wsum = None
            wtot = None
            avail = 0
            has_conf_any = False

            for r in roots_use:
                mp = r / split / clip / "masks" / name
                if not mp.exists():
                    continue
                m = _read_mask(mp)
                if m is None:
                    continue

                cp = r / split / clip / "conf" / name
                c = _read_conf(cp) if cp.exists() else None
                if c is None:
                    c = np.ones((size, size), np.float32)
                else:
                    has_conf_any = True

                if count is None:
                    count = m.astype(np.uint16)
                    wsum = (c * m.astype(np.float32))
                    wtot = c.astype(np.float32)
                else:
                    count += m
                    wsum += c * m.astype(np.float32)
                    wtot += c
                avail += 1

            if count is None or avail < vote_min_seeds:
                skipped_missing += 1
                continue

            if has_conf_any:
                used_conf_frames += 1

            soft = wsum / (wtot + 1e-6)

            thr_frac = 0.5
            if vote_min > 0:
                thr_frac = min(1.0, float(vote_min) / float(max(1, avail)))

            core = (soft >= thr_frac)
            if vote_min > 0:
                core = core & (count >= vote_min)

            merged = np.zeros((size, size), np.float32)
            merged[core] = 1.0

            if uncert_val > 0:
                unc = (~core) & (soft >= uncert_val)
            else:
                unc = (~core) & (soft > 0)
            merged[unc] = soft[unc]

            merged_u8 = np.clip(np.round(merged * 255.0), 0, 255).astype(np.uint8)
            status = _write_pair(name, merged_u8)
            if status == "ok":
                kept += 1
            elif status == "small":
                skipped_small += 1
            else:
                skipped_missing += 1
            continue

        # strategy: union
        merged = None
        for r in roots:
            mp = r / split / clip / "masks" / name
            if not mp.exists():
                continue
            m = _read_mask(mp)
            if m is None:
                continue
            m = m * 255
            merged = m if merged is None else np.maximum(merged, m)

        if merged is None:
            skipped_missing += 1
            continue

        status = _write_pair(name, merged.astype(np.uint8))
        if status == "ok":
            kept += 1
        elif status == "small":
            skipped_small += 1
        else:
            skipped_missing += 1

    print(
        f"[OK] merged masks/frames kept={kept} skipped_eval={skipped_eval} "
        f"skipped_small={skipped_small} skipped_missing={skipped_missing} -> {dst_clip}"
    )
    print(f"[INFO] forced NEG frames = {forced_neg} (from NEG_LIST)")
    print(f"[INFO] conf-weighted vote used on {used_conf_frames}/{len(names)} frames (need SAVE_CONF=1 in filter to benefit).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_multi")
    ap.add_argument("split")
    ap.add_argument("clip")
    ap.add_argument("size", type=int)
    ap.add_argument("frame_dir")
    ap.add_argument("minpix", type=int)
    ap.add_argument("roots", nargs="+", help="pseudolabel roots, e.g. outputs/pseudolabels_clean_seed_0001")
    args = ap.parse_args()

    argv = [
        "merge_seeds.py",
        args.out_multi,
        args.split,
        args.clip,
        str(args.size),
        args.frame_dir,
        str(args.minpix),
    ] + list(args.roots)
    _run_with_sysargv(argv)


if __name__ == "__main__":
    main()
