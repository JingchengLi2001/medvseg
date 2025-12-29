import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from medvseg.utils.flow import warp_mask, fb_consistency_conf, iou_weighted

def read_mask01(p: Path, resize_to: int | None = None):
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if resize_to is not None:
        m = cv2.resize(m, (resize_to, resize_to), interpolation=cv2.INTER_NEAREST)
    m = (m > 0).astype(np.float32)
    return torch.from_numpy(m)[None, None]

def main(pred_root: str, flow_root: str, output_root: str,
         iou_th: float = 0.7, resize: int | None = None,
         conf_sigma: float = 2.0, conf_thr: float = 0.6,
         min_conf: float = 0.0, min_conf_pct: float = 0.0,
         save_conf: int = 0, flow_p95_th: float = 0.0, flow_mean_th: float = 0.0):
    pred_root = Path(pred_root)
    flow_root = Path(flow_root)
    output_root = Path(output_root)

    for split in sorted(pred_root.iterdir()):
        if not split.is_dir() or split.name == "flows":
            continue
        for clip in sorted(split.iterdir()):
            fdir = clip / "frames"
            mdir = clip / "masks"
            if not fdir.exists() or not mdir.exists():
                continue

            out = output_root / split.name / clip.name
            (out / "frames").mkdir(parents=True, exist_ok=True)
            (out / "masks").mkdir(parents=True, exist_ok=True)
            if save_conf:
                (out / "conf").mkdir(parents=True, exist_ok=True)

            frames = sorted(fdir.glob("*.png"))
            flows_dir = flow_root / split.name / clip.name

            # 1) copy frames (optional resize)
            for f in frames:
                img = cv2.imread(str(f))
                if img is None:
                    continue
                if resize is not None:
                    img = cv2.resize(img, (resize, resize))
                cv2.imwrite(str(out / "frames" / f.name), img)

            kept = 0
            for i in range(len(frames) - 1):
                t = frames[i]
                tp1 = frames[i + 1]

                Mt = read_mask01(mdir / t.name, resize)
                Mtp1 = read_mask01(mdir / tp1.name, resize)
                if Mt is None or Mtp1 is None:
                    continue

                fwd_path = flows_dir / f"{t.stem}_to_{tp1.stem}.npy"
                bwd_path = flows_dir / f"{tp1.stem}_to_{t.stem}.npy"
                if not fwd_path.exists():
                    continue
                flow_fwd = np.load(str(fwd_path))
                if bwd_path.exists():
                    flow_bwd = np.load(str(bwd_path))
                else:
                    flow_bwd = -flow_fwd
                # ----- flow magnitude gate (motion break) -----
                if flow_p95_th > 0.0 or flow_mean_th > 0.0:
                    mag = np.sqrt(flow_fwd[..., 0] ** 2 + flow_fwd[..., 1] ** 2).astype(np.float32)
                    p95 = float(np.percentile(mag, 95))
                    mean = float(mag.mean())
                    if (flow_p95_th > 0.0 and p95 > flow_p95_th) or (flow_mean_th > 0.0 and mean > flow_mean_th):
                        # too much motion -> drop this frame
                        continue


                # warp M_{t+1} back to t (target=t, source=t+1)
                Mtp1_warp_to_t = warp_mask(Mtp1, flow_fwd)

                Mt_bin = (Mt > 0.5).float()
                Mw_bin = (Mtp1_warp_to_t > 0.5).float()
                roi = ((Mt_bin + Mw_bin) > 0).float()
                if roi.sum().item() < 1:
                    continue

                conf_np = fb_consistency_conf(flow_fwd, flow_bwd, sigma=conf_sigma, device="cpu")
                conf = torch.from_numpy(conf_np)[None, None].float()

                roi_np = roi[0, 0].numpy() > 0
                roi_cnt = int(roi_np.sum())
                if roi_cnt < 1:
                    continue
                roi_conf_mean = float((conf * roi).sum().item() / (roi.sum().item() + 1e-6))
                roi_conf_pct = float(((conf_np > conf_thr) & roi_np).sum() / roi_cnt)

                # break-point gate
                if (roi_conf_mean < min_conf) or (roi_conf_pct < min_conf_pct):
                    if save_conf:
                        vis = (conf_np * 255).clip(0, 255).astype("uint8")
                        cv2.imwrite(str(out / "conf" / t.name), vis)
                    continue

                w = conf * roi
                iou = iou_weighted(Mt_bin, Mw_bin, w)
                if iou >= iou_th:
                    cv2.imwrite(str(out / "masks" / t.name),
                                (Mt[0, 0].numpy() * 255).astype("uint8"))
                    kept += 1

                if save_conf:
                    vis = (conf_np * 255).clip(0, 255).astype("uint8")
                    cv2.imwrite(str(out / "conf" / t.name), vis)

            print("Filtered ->", out, "kept=", kept)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--flow-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--iou-th", type=float, default=0.7)
    ap.add_argument("--resize", type=int, default=None)
    ap.add_argument("--conf-sigma", type=float, default=2.0)
    ap.add_argument("--conf-thr", type=float, default=0.6)
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--min-conf-pct", type=float, default=0.0)
    ap.add_argument("--save-conf", type=int, default=0)
    ap.add_argument("--flow-p95-th", type=float, default=0.0, help="drop frame if flow magnitude p95 exceeds this (>0 enables)")
    ap.add_argument("--flow-mean-th", type=float, default=0.0, help="drop frame if flow magnitude mean exceeds this (>0 enables)")
    args = ap.parse_args()
    main(**vars(args))