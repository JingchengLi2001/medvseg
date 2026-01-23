
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from medvseg.models.student_unet import StudentUNet
from medvseg.utils.motion_gate import GateConfig, MotionGate, auto_calibrate, motion_metrics

def _load_model(ckpt: str, device: str):
    net = StudentUNet("resnet34", 3, 1)
    state = torch.load(ckpt, map_location=device)
    # support both {model: ...} and raw state_dict
    if isinstance(state, dict) and "model" in state:
        net.load_state_dict(state["model"])
    else:
        net.load_state_dict(state)
    net.to(device).eval()
    return net

def _iter_frames(cap: cv2.VideoCapture):
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame

def _calibrate_from_video(video: str, cfg: GateConfig) -> dict:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    dt = max(1, int(round((fps if fps > 0 else 25.0) * cfg.dt_time)))
    # read up to calib_seconds (or full video if shorter)
    max_frames = int((fps if fps > 0 else 25.0) * cfg.calib_seconds) if cfg.calib_seconds > 0 else 10**9

    buf = []
    metrics = []
    for i, frame in enumerate(_iter_frames(cap)):
        buf.append(frame)
        if len(buf) > dt + 1:
            prev = buf[-dt-1]
            curr = buf[-1]
            metrics.append(motion_metrics(prev, curr, size=cfg.size, conf_sigma=cfg.conf_sigma, conf_thr=cfg.conf_thr))
        if i + 1 >= max_frames:
            break

    cap.release()
    return auto_calibrate(metrics, stable_quantile=cfg.stable_quantile, k_mad=cfg.k_mad)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="best.ckpt or state dict")
    ap.add_argument("--video-src", required=True, help="input mp4")
    ap.add_argument("--save", required=True, help="output mp4")
    ap.add_argument("--resize", type=int, default=512, help="seg model input size")
    ap.add_argument("--thr", type=float, default=0.5, help="segmentation threshold")
    ap.add_argument("--smooth", type=float, default=0.5, help="temporal smoothing factor [0..1]")
    ap.add_argument("--half", type=int, default=1, help="use fp16 on cuda if 1")
    ap.add_argument("--min-area", type=int, default=50, help="drop tiny masks")
    ap.add_argument("--save-masks", type=str, default="", help="(optional) save gated binary masks per frame to this dir")
    ap.add_argument("--save-metrics", type=str, default="", help="(optional) save per-frame gate metrics to this CSV path")
    # motion gate
    ap.add_argument("--gate", type=int, default=1, help="enable motion-break gate")
    ap.add_argument("--gate-auto", type=int, default=1, help="auto-calibrate per-video thresholds (MAD)")
    ap.add_argument("--gate-calib-seconds", type=float, default=3.0)
    ap.add_argument("--gate-stable-quantile", type=float, default=0.6)
    ap.add_argument("--gate-k-mad", type=float, default=4.0)
    ap.add_argument("--gate-k-on", type=int, default=3)
    ap.add_argument("--gate-on-ratio", type=float, default=1.0, help="hysteresis: flow_p95 must be < flow_p95_th * ratio to re-enable")
    ap.add_argument("--gate-cooldown", type=int, default=0, help="hold disabled for N frames after each break")
    ap.add_argument("--gate-cooldown-high", type=int, default=0, help="optional longer cooldown for severe breaks")
    ap.add_argument("--gate-cooldown-high-mult", type=float, default=2.0, help="severe if flow_p95 > flow_p95_th * mult")
    ap.add_argument("--gate-size", type=int, default=256)
    ap.add_argument("--gate-conf-sigma", type=float, default=2.0)
    ap.add_argument("--gate-conf-thr", type=float, default=0.6)
    ap.add_argument("--gate-dt-time", type=float, default=1.0/25.0, help="fps-normalized time step in seconds")
    # optional manual override thresholds
    ap.add_argument("--flow-p95-th", type=float, default=None)
    ap.add_argument("--conf-mean-th", type=float, default=None)
    ap.add_argument("--conf-pct-th", type=float, default=None)
    ap.add_argument("--blur-var-th", type=float, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = bool(args.half) and device == "cuda"

    net = _load_model(args.model, device)

    cap = cv2.VideoCapture(args.video_src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video_src}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(save_path), fourcc, float(fps), (w, h))

    gate_cfg = GateConfig(
        size=args.gate_size,
        conf_sigma=args.gate_conf_sigma,
        conf_thr=args.gate_conf_thr,
        k_on=max(1, args.gate_k_on),
        on_ratio=float(args.gate_on_ratio),
        cooldown=max(0, int(args.gate_cooldown)),
        cooldown_high=max(0, int(args.gate_cooldown_high)),
        cooldown_high_mult=float(args.gate_cooldown_high_mult),
        auto=bool(args.gate_auto),
        calib_seconds=args.gate_calib_seconds,
        stable_quantile=args.gate_stable_quantile,
        k_mad=args.gate_k_mad,
        dt_time=args.gate_dt_time,
    )
    gate = MotionGate(gate_cfg)

    # thresholds: either auto or manual or fallback defaults
    th = None
    if args.gate and gate_cfg.auto:
        try:
            th = _calibrate_from_video(args.video_src, gate_cfg)
        except Exception as e:
            print("[WARN] auto-calibrate failed:", repr(e))
            th = None

    # manual override
    if th is None:
        th = {
            "flow_p95_th": 8.0,
            "conf_mean_th": 0.55,
            "conf_pct_th": 0.15,
            "blur_var_th": 15.0,
        }
    if args.flow_p95_th is not None: th["flow_p95_th"] = float(args.flow_p95_th)
    if args.conf_mean_th is not None: th["conf_mean_th"] = float(args.conf_mean_th)
    if args.conf_pct_th is not None: th["conf_pct_th"] = float(args.conf_pct_th)
    if args.blur_var_th is not None: th["blur_var_th"] = float(args.blur_var_th)
    gate.set_thresholds(th)

    print("[INFO] Motion gate:", "ON" if args.gate else "OFF")
    if args.gate:
        print("[INFO] Gate thresholds:", {k: float(v) for k, v in th.items()})
        print("[INFO] Gate cfg:", {
            "dt_time": gate_cfg.dt_time,
            "dt_frames": gate.dt_frames(fps),
            "k_on": gate_cfg.k_on,
            "size": gate_cfg.size,
            "conf_sigma": gate_cfg.conf_sigma,
            "conf_thr": gate_cfg.conf_thr,
        })


    mask_dir = Path(args.save_masks) if args.save_masks else None
    if mask_dir is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = Path(args.save_metrics) if args.save_metrics else None
    metrics_rows = []  # list[dict]
    prev_for_metrics = None  # used when args.gate=0 but save-metrics is set
    prev_mask = None
    enabled_prev = False

    with torch.no_grad():
        for frame_idx, frame_bgr in enumerate(_iter_frames(cap)):
            # motion break?
            if args.gate:
                is_break, met = gate.step(frame_bgr, fps)
                enabled = gate.enabled
                if is_break:
                    # reset temporal smoothing when break happens
                    prev_mask = None
                # if currently disabled -> force empty mask
                if not enabled:
                    mask = np.zeros((h, w), np.uint8)
                    prev_mask = None  # keep strict: no trailing
                else:
                    mask = None  # to be computed below
            else:
                enabled = True
                is_break = False
                met = None
                # optionally compute metrics even when gate is off
                if metrics_csv is not None:
                    if prev_for_metrics is not None:
                        try:
                            met = motion_metrics(prev_for_metrics, frame_bgr,
                                                 size=int(args.gate_size),
                                                 conf_sigma=float(args.gate_conf_sigma),
                                                 conf_thr=float(args.gate_conf_thr))
                        except Exception:
                            met = None
                    prev_for_metrics = frame_bgr.copy()
                mask = None

            # record metrics (if requested)
            if metrics_csv is not None:
                row = {
                    "frame_idx": int(frame_idx),
                    "enabled": int(1 if enabled else 0),
                    "break": int(1 if is_break else 0),
                }
                if isinstance(met, dict):
                    row.update({k: float(v) for k, v in met.items()})
                metrics_rows.append(row)



            if mask is None:
                inp = cv2.resize(frame_bgr, (args.resize, args.resize), interpolation=cv2.INTER_LINEAR)
                inp_rgb = inp[:, :, ::-1].copy()
                x = torch.from_numpy(inp_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                x = x.to(device)
                if half:
                    x = x.half()
                with torch.autocast(device_type="cuda", enabled=half):
                    logits = net(x)
                    prob = torch.sigmoid(logits)[0, 0].float().cpu().numpy()

                prob_up = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
                if prev_mask is None:
                    sm = prob_up
                else:
                    sm = args.smooth * prev_mask + (1.0 - args.smooth) * prob_up
                prev_mask = sm
                mask = (sm >= args.thr).astype(np.uint8)


                if int(mask.sum()) < int(args.min_area):
                    mask[:] = 0

            # save gated mask (optional)
            if mask_dir is not None:
                out_mask = (mask.astype(np.uint8) * 255)
                cv2.imwrite(str(mask_dir / f"{int(frame_idx):05d}.png"), out_mask)
            overlay = frame_bgr.copy()
            overlay[mask > 0] = (0, 0, 255)
            out = cv2.addWeighted(frame_bgr, 0.65, overlay, 0.35, 0)
            writer.write(out)

    cap.release()
    writer.release()
        # write metrics csv (optional)
    if metrics_csv is not None and metrics_rows:
        import csv
        fieldnames = ["frame_idx", "enabled", "break",
                      "flow_mean", "flow_p95", "conf_mean", "conf_pct", "blur_var"]
        extra = sorted({k for r in metrics_rows for k in r.keys()} - set(fieldnames))
        fieldnames = fieldnames + extra
        metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        with metrics_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in metrics_rows:
                w.writerow(r)
        print("[OK] Saved gate metrics ->", str(metrics_csv))


if __name__ == "__main__":
    main()
