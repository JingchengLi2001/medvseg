
"""
Endoscopy one-click pipeline (maintainable replacement for the long run.sh).

Goal:
1) Stable / clear lesion -> segment precisely (train + overlay).
2) Shaky / large motion -> gate OFF segmentation (overlay).

This pipeline keeps the original stage order:
  propagate -> filter -> merge -> manual_val -> train -> auto_th -> eval -> overlay

It reads most settings from environment variables to stay compatible with old usage:
  DATASET_ROOT, SPLIT, CLIP, RESIZE, IOU_TH, EPOCHS, BATCH_SIZE, LR,
  MULTI_SEED, SEED_LIST / SEED_NAME,
  MERGE_STRATEGY, VOTE_MIN, VOTE_MIN_SEEDS, MERGE_TOPK, UNCERT_VAL,
  CONF_SIGMA, CONF_THR, MIN_CONF, MIN_CONF_PCT, SAVE_CONF,
  VIDEO_SRC, OUT_VIDEO,
  FLOW_GATE_AUTO (overlay), etc.

Usage:
  DATASET_ROOT=data/MED SPLIT=test_easy_seen CLIP=clip_0002 MULTI_SEED=0 SEED_NAME=0280.png VIDEO_SRC=/path/v.mp4 ./run.sh
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

from medvseg.utils.motion_gate import robust_threshold_high

def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)

def _envi(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except Exception:
        return default

def _envf(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except Exception:
        return default

def _run(cmd: List[str], log_path: Path | None = None):
    cmd_str = " ".join(cmd)
    print("\n[CMD]", cmd_str)
    if log_path is None:
        subprocess.run(cmd, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)

def _count_masks(root: Path) -> int:
    return len(list(root.glob("**/masks/*.png")))

def _prune_to_one_clip(root: Path, split: str, clip: str):
    # keep only {split} and flows
    if not root.exists():
        return
    for p in root.iterdir():
        if p.is_dir() and p.name not in ["flows", split]:
            shutil.rmtree(p, ignore_errors=True)
    spl = root / split
    if spl.exists():
        for p in spl.iterdir():
            if p.is_dir() and p.name != clip:
                shutil.rmtree(p, ignore_errors=True)
    flows = root / "flows"
    if flows.exists():
        for p in flows.iterdir():
            if p.is_dir() and p.name != split:
                shutil.rmtree(p, ignore_errors=True)
        spl2 = flows / split
        if spl2.exists():
            for p in spl2.iterdir():
                if p.is_dir() and p.name != clip:
                    shutil.rmtree(p, ignore_errors=True)

def _drop_empty_masks_inplace(out_clean: Path, split: str, clip: str, minpix: int):
    mdir = out_clean / split / clip / "masks"
    fdir = out_clean / split / clip / "frames"
    if not mdir.exists():
        return
    kept = 0
    removed = 0
    for mp in sorted(mdir.glob("*.png")):
        import cv2
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            removed += 1
            mp.unlink(missing_ok=True)
            continue
        if int((m > 0).sum()) < int(minpix):
            removed += 1
            mp.unlink(missing_ok=True)
            fp = fdir / mp.name
            if fp.exists():
                fp.unlink(missing_ok=True)
        else:
            kept += 1
    print(f"[OK] drop-empty: kept={kept} removed={removed} (minpix={minpix})")

def _estimate_flow_p95_threshold(flow_dir: Path) -> float | None:
    """
    Estimate a per-video flow_p95 threshold from saved flow npy files (stable subset + MAD).
    Returns None if flows missing.
    """
    if not flow_dir.exists():
        return None
    import numpy as np
    vals = []
    for p in sorted(flow_dir.glob("**/*.npy")):
        # only forward flows (heuristic): filenames containing "_to_"
        if "_to_" not in p.name:
            continue
        try:
            flow = np.load(p)
            if flow.ndim != 3 or flow.shape[-1] != 2:
                continue
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)
            vals.append(float(np.percentile(mag, 95)))
        except Exception:
            continue
        if len(vals) > 800:
            break
    if len(vals) < 20:
        return None
    import numpy as np
    arr = np.array(vals, dtype=np.float32)
    # stable subset = lower 60% motion
    q = float(np.quantile(arr, 0.6))
    stable = arr[arr <= q]
    if stable.size < 10:
        stable = arr
    return float(max(0.5, robust_threshold_high(stable, k=4.0)))

def main():
    here = Path(__file__).resolve().parents[2]
    os.chdir(here)

    dataset_root = Path(_env("DATASET_ROOT", "data/MED"))
    split = _env("SPLIT", "test_easy_seen")
    clip = _env("CLIP", "clip_0001")
    resize = _envi("RESIZE", 512)

    iou_th = _envf("IOU_TH", 0.4)
    epochs = _envi("EPOCHS", 10)
    batch_size = _envi("BATCH_SIZE", 4)
    lr = _envf("LR", 1e-3)

    smooth = _envf("SMOOTH", 0.5)
    pred_th = _envf("PRED_TH", 0.5)
    half = _envi("HALF", 1)
    minpix = _envi("MIN_MASK_PIXELS", 50)

    multi_seed = _envi("MULTI_SEED", 0)
    seed_name = _env("SEED_NAME", "")
    seed_list = _env("SEED_LIST", "").split()

    merge_strategy = _env("MERGE_STRATEGY", "vote")
    vote_min = _envi("VOTE_MIN", 0)
    vote_min_seeds = _envi("VOTE_MIN_SEEDS", 1)
    merge_topk = _envi("MERGE_TOPK", 3)
    uncert_val = _envf("UNCERT_VAL", 0.30)

    conf_sigma = _envf("CONF_SIGMA", 2.0)
    conf_thr = _envf("CONF_THR", 0.6)
    min_conf = _envf("MIN_CONF", 0.25)
    min_conf_pct = _envf("MIN_CONF_PCT", 0.15)
    save_conf = _envi("SAVE_CONF", 1)

    video_src = _env("VIDEO_SRC", "")
    out_video = _env("OUT_VIDEO", "outputs/seg_overlay.mp4")

    clean = _envi("CLEAN", 1)
    if clean == 1:
        print("[INFO] Cleaning old outputs (only outputs/)...")
        for pat in ["outputs/xmem_raw_seed_*", "outputs/pseudolabels_clean_seed_*",
                    "outputs/pseudolabels_clean_multi", "outputs/runs/unet_r34",
                    "outputs/val_manual", out_video]:
            for p in Path(".").glob(pat):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)

    # build tmp images root so propagate only sees one clip (same trick as old run.sh)
    tmp_images_root = Path(f"outputs/_tmp_images_root_{split}_{clip}")
    if tmp_images_root.exists():
        shutil.rmtree(tmp_images_root, ignore_errors=True)
    (tmp_images_root / split).mkdir(parents=True, exist_ok=True)
    target = dataset_root / split / clip
    link = tmp_images_root / split / clip
    if link.exists():
        link.unlink()
    link.symlink_to(target.resolve())

    # seeds
    if multi_seed == 1:
        seeds = seed_list
    else:
        seeds = [seed_name] if seed_name else seed_list
    if not seeds:
        raise SystemExit("[ERR] No seeds. Set SEED_NAME or SEED_LIST (and MULTI_SEED=1).")

    logs_dir = Path("outputs/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    clean_roots: List[str] = []
    for s in seeds:
        tag = s.replace(".png", "")
        out_raw = Path(f"outputs/xmem_raw_seed_{tag}")
        out_clean = Path(f"outputs/pseudolabels_clean_seed_{tag}")
        clean_roots.append(str(out_clean))

        if out_raw.exists(): shutil.rmtree(out_raw, ignore_errors=True)
        if out_clean.exists(): shutil.rmtree(out_clean, ignore_errors=True)

        print(f"\n[INFO] ===== Seed {s} (tag={tag}) =====")
        _run([sys.executable, "-m", "medvseg.engines.propagate_baseline",
              "--images-root", str(tmp_images_root),
              "--output-root", str(out_raw),
              "--resize", str(resize),
              "--seed-name", s],
             log_path=logs_dir / f"propagate_seed_{tag}.log")

        _prune_to_one_clip(out_raw, split, clip)

        # auto estimate flow threshold for filtering (optional, helps remove shaky frames from TRAIN)
        flow_p95_th = _estimate_flow_p95_threshold(out_raw / "flows")
        cmd = [sys.executable, "-m", "medvseg.engines.filter_pseudolabels",
               "--pred-root", str(out_raw),
               "--flow-root", str(out_raw / "flows"),
               "--output-root", str(out_clean),
               "--iou-th", str(iou_th),
               "--conf-sigma", str(conf_sigma),
               "--conf-thr", str(conf_thr),
               "--min-conf", str(min_conf),
               "--min-conf-pct", str(min_conf_pct),
               "--save-conf", str(save_conf)]
        if flow_p95_th is not None:
            cmd += ["--flow-p95-th", str(flow_p95_th)]
            print("[INFO] TRAIN flow gate (auto) --flow-p95-th =", flow_p95_th)
        _run(cmd, log_path=logs_dir / f"filter_seed_{tag}.log")

        _drop_empty_masks_inplace(out_clean, split, clip, minpix=minpix)

        kept = len(list((out_clean / split / clip / "masks").glob("*.png")))
        print(f"[INFO] Seed {s} final kept masks = {kept}")

    # merge
    out_multi = Path("outputs/pseudolabels_clean_multi")
    frame_dir = dataset_root / split / clip / "frames"
    env = os.environ.copy()
    env["MERGE_STRATEGY"] = merge_strategy
    env["VOTE_MIN"] = str(vote_min)
    env["VOTE_MIN_SEEDS"] = str(vote_min_seeds)
    env["MERGE_TOPK"] = str(merge_topk)
    env["UNCERT_VAL"] = str(uncert_val)
    env["MERGE_SEEDS"] = " ".join(seeds)

    print(f"\n[INFO] Merging seeds -> {out_multi}/{split}/{clip} (strategy={merge_strategy})")
    subprocess.run([sys.executable, "-m", "medvseg.engines.merge_seeds",
                    str(out_multi), split, clip, str(resize), str(frame_dir), str(minpix)] + clean_roots,
                   check=True, env=env)

    train_root = out_multi / split

    # manual val set
    val_manual_root = Path("outputs/val_manual")
    tune_dataset_root = dataset_root  # will switch to val_manual_root if manual set is available
    subprocess.run([sys.executable, "-m", "medvseg.engines.make_manual_val",
                    str(dataset_root), split, clip, str(resize), str(val_manual_root), _env("EVAL_ONLY_LIST","")] + seeds,
                   check=True, env=os.environ.copy())
    val_root = train_root
    if _count_masks(val_manual_root / split) > 0:
        val_root = val_manual_root / split
        print(f"[INFO] Using MANUAL val set: {val_root}/{clip}")
        tune_dataset_root = val_manual_root

    else:
        print("[WARN] Manual val set empty -> fallback to pseudo val (same as train).")

    # train
    save_dir = Path("outputs/runs/unet_r34")
    subprocess.run([sys.executable, "-m", "medvseg.engines.train_student",
                    "--data-root", str(train_root),
                    "--val-root", str(val_root),
                    "--save-dir", str(save_dir),
                    "--epochs", str(epochs),
                    "--batch-size", str(batch_size),
                    "--lr", str(lr)],
                   check=True)

    ckpt = save_dir / "best.ckpt"
    if not ckpt.exists():
        raise SystemExit("[ERR] best.ckpt not found after training.")

    # auto threshold (avoid manual tuning across different videos)
    auto_th_enable = _envi("AUTO_TH_ENABLE", 1)
    if auto_th_enable == 1:
        print("[INFO] Auto-tuning PRED_TH on MANUAL frames...")
        # capture stdout from module
        out = subprocess.check_output([sys.executable, "-m", "medvseg.engines.auto_tune_threshold",
                                       str(tune_dataset_root), split, clip, str(ckpt), str(resize), _env("EVAL_ONLY_LIST","")] + seeds,
                                      env=os.environ.copy(), text=True)
        out = out.strip()
        if out:
            try:
                pred_th = float(out.splitlines()[-1])
                print("[INFO] Auto PRED_TH =", pred_th)
            except Exception:
                print("[WARN] Failed to parse AUTO_TH output:", out)

    # eval on manual seeds (optional)
    if _envi("EVAL_MANUAL_SEEDS", 1) == 1:
        seed_dir = dataset_root / split / clip / "masks"
        eval_frame_dir = frame_dir
        eval_mask_dir = seed_dir
        if tune_dataset_root == val_manual_root:
            eval_frame_dir = val_manual_root / split / clip / "frames"
            eval_mask_dir = val_manual_root / split / clip / "masks"
        if eval_mask_dir.exists():
            subprocess.run([sys.executable, "-m", "medvseg.engines.eval_manual_seeds",
                            str(eval_frame_dir), str(eval_mask_dir), str(ckpt), str(resize), str(pred_th)] + seeds,
                           check=True)

    # overlay
    if video_src:
        print("[INFO] Overlay video with motion-break gate...")
        # motion gate options (auto by default)
        gate = _envi("FLOW_GATE", 1)
        gate_auto = _envi("FLOW_GATE_AUTO", 1)
        gate_calib_seconds = _envf("FLOW_GATE_CALIB_SECONDS", 3.0)
        gate_k_on = _envi("FLOW_GATE_K_ON", 3)
        gate_on_ratio = _envf("FLOW_GATE_ON_RATIO", 1.0)
        gate_cooldown = _envi("FLOW_GATE_COOLDOWN", 0)
        gate_cooldown_high = _envf("FLOW_GATE_COOLDOWN_HIGH", 10.0)
        gate_cooldown_high_mult = _envf("FLOW_GATE_COOLDOWN_HIGH_MULT", 2.0)
        gate_size = _envi("FLOW_GATE_SIZE", 256)
        gate_conf_sigma = _envf("FLOW_GATE_CONF_SIGMA", 2.0)
        gate_conf_thr = _envf("FLOW_GATE_CONF_THR", 0.6)
        gate_dt_time = _envf("FLOW_GATE_DT_TIME", 1.0/25.0)
        gate_save_masks = _env("GATE_SAVE_MASKS", "")
        gate_save_metrics = _env("GATE_SAVE_METRICS", "")
        eval_gate = _envi("EVAL_GATE", 0)


        cmd = [sys.executable, "-m", "medvseg.engines.overlay_video",
                        "--model", str(ckpt),
                        "--video-src", str(video_src),
                        "--save", str(out_video),
                        "--resize", str(resize),
                        "--thr", str(pred_th),
                        "--smooth", str(smooth),
                        "--half", str(half),
                        "--min-area", str(minpix),
                        "--gate", str(gate),
                        "--gate-auto", str(gate_auto),
                        "--gate-calib-seconds", str(gate_calib_seconds),
                        "--gate-k-on", str(gate_k_on),
                        "--gate-on-ratio", str(gate_on_ratio),
                        "--gate-cooldown", str(gate_cooldown),
                        "--gate-cooldown-high", str(gate_cooldown_high),
                        "--gate-cooldown-high-mult", str(gate_cooldown_high_mult),
                        "--gate-size", str(gate_size),
                        "--gate-conf-sigma", str(gate_conf_sigma),
                        "--gate-conf-thr", str(gate_conf_thr),
                        "--gate-dt-time", str(gate_dt_time)]
        if gate_save_masks:
            cmd += ["--save-masks", str(gate_save_masks)]
        if gate_save_metrics:
            cmd += ["--save-metrics", str(gate_save_metrics)]
        subprocess.run(cmd, check=True)

        # optional gate summary
        if eval_gate == 1 and gate_save_metrics:
            try:
                gcmd = [sys.executable, "-m", "medvseg.engines.eval_gate_metrics",
                        str(gate_save_metrics)]
                if gate_save_masks:
                    gcmd += ["--masks-dir", str(gate_save_masks)]
                subprocess.run(gcmd, check=False)
            except Exception:
                pass

    else:
        print("[WARN] VIDEO_SRC not set -> skip overlay video.")

    print("\n[OK] Pipeline finished.")
    print(" - best.ckpt:", ckpt)
    if video_src:
        print(" - overlay:", out_video)

if __name__ == "__main__":
    main()
