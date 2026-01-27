import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from medvseg.utils.common import ensure_dir
from medvseg.utils.flow import warp_mask, iou_binary
from medvseg.engines import propagate_baseline


def _read_mask01(p: Path, resize: int) -> torch.Tensor:
    """
    Read a grayscale mask and return torch float (1,1,H,W) in {0,1}.
    Compatible with your existing masks (0/255 or uncertainty-encoded 0..255).
    """
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {p}")
    if resize:
        m = cv2.resize(m, (resize, resize), interpolation=cv2.INTER_NEAREST)
    m01 = (m > 0).astype(np.float32)
    return torch.from_numpy(m01)[None, None]


def _read_frame_tensor(p: Path) -> torch.Tensor:
    """
    Read RGB frame as torch float (1,3,H,W) in [0,1].
    Matches your training input style (no mean/std normalization).
    """
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read frame: {p}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    return x.unsqueeze(0)


def _load_student_model(ckpt: Path, device: str) -> torch.nn.Module:
    # Lazy import to avoid import-time failures when backend=baseline.
    from medvseg.models.student_unet import StudentUNet

    model = StudentUNet("resnet34", 3, 1)
    state = torch.load(str(ckpt), map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _predict_prob(model: torch.nn.Module, x: torch.Tensor, device: str) -> np.ndarray:
    x = x.to(device)
    logit = model(x)
    prob = torch.sigmoid(logit)[0, 0].detach().cpu().numpy().astype(np.float32)
    return prob


def _flow_path(flow_dir: Path, tgt_name: str, src_name: str) -> Path:
    tgt = Path(tgt_name).stem
    src = Path(src_name).stem
    return flow_dir / f"{tgt}_to_{src}.npy"


def _select_cc_by_iou(bin01: np.ndarray, guide01: np.ndarray, min_area: int) -> np.ndarray:
    """
    Select a connected component from bin01 that best overlaps guide01.
    Returns uint8 {0,1} mask. If no component passes filters, returns all zeros.
    """
    if bin01.sum() == 0:
        return np.zeros_like(bin01, dtype=np.uint8)

    num, lab, stats, _ = cv2.connectedComponentsWithStats(bin01.astype(np.uint8), connectivity=8)
    best_iou = -1.0
    best = None

    guide_t = torch.from_numpy(guide01.astype(np.float32))[None, None]
    for k in range(1, num):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if min_area > 0 and area < min_area:
            continue
        comp = (lab == k).astype(np.float32)
        iou = iou_binary(torch.from_numpy(comp)[None, None], guide_t)
        if iou > best_iou:
            best_iou = iou
            best = (lab == k).astype(np.uint8)

    if best is None:
        return np.zeros_like(bin01, dtype=np.uint8)
    return best.astype(np.uint8)


def _write_mask(out_p: Path, m01: np.ndarray) -> None:
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), (m01.astype(np.uint8) * 255))


def _modeltrack_clip(
    frames_out_dir: Path,
    masks_in_dir: Path,
    masks_out_dir: Path,
    flow_dir: Path,
    seed_name: str,
    resize: int,
    teacher_ckpt: Path,
    thr: float,
    min_area: int,
    iou_track_min: float,
) -> None:
    frames = sorted(frames_out_dir.glob("*.png"))
    if not frames:
        return

    seed_mask_path = masks_in_dir / seed_name
    if not seed_mask_path.exists():
        raise RuntimeError(f"seed_name={seed_name} but file not found: {seed_mask_path}")

    # find seed index
    seed_idx: Optional[int] = None
    for i, fp in enumerate(frames):
        if fp.name == seed_name:
            seed_idx = i
            break
    if seed_idx is None:
        raise RuntimeError(
            f"Seed mask name must match a frame name.\n"
            f"mask={seed_name} but frame not found under frames/: {frames_out_dir}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_student_model(teacher_ckpt, device)

    # init track with seed mask
    track = _read_mask01(seed_mask_path, resize).to(device)  # 1,1,H,W
    seed_bin = (track[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
    _write_mask(masks_out_dir / seed_name, seed_bin)

    # forward (seed -> end): warp using backward flow (curr -> prev)
    for i in range(seed_idx + 1, len(frames)):
        curr = frames[i]
        prev = frames[i - 1]
        fp = _flow_path(flow_dir, tgt_name=curr.name, src_name=prev.name)
        if not fp.exists():
            raise RuntimeError(f"Missing flow file: {fp}")
        flow = np.load(str(fp)).astype(np.float32)  # (H,W,2), curr -> prev
        track = warp_mask(track, flow)
        guide01 = (track[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)

        prob = _predict_prob(model, _read_frame_tensor(curr), device)
        bin01 = (prob >= float(thr)).astype(np.uint8)
        sel01 = _select_cc_by_iou(bin01, guide01, min_area=min_area)

        if iou_track_min > 0:
            iou_val = iou_binary(
                torch.from_numpy(sel01.astype(np.float32))[None, None],
                torch.from_numpy(guide01.astype(np.float32))[None, None],
            )
            if iou_val < float(iou_track_min):
                sel01[:] = 0

        _write_mask(masks_out_dir / curr.name, sel01)

        if sel01.sum() > 0:
            track = torch.from_numpy(sel01.astype(np.float32))[None, None].to(device)

    # backward (seed -> start): warp using forward flow (curr -> next), because source is next frame
    track = _read_mask01(seed_mask_path, resize).to(device)
    for i in range(seed_idx - 1, -1, -1):
        curr = frames[i]
        nxt = frames[i + 1]
        fp = _flow_path(flow_dir, tgt_name=curr.name, src_name=nxt.name)
        if not fp.exists():
            raise RuntimeError(f"Missing flow file: {fp}")
        flow = np.load(str(fp)).astype(np.float32)  # curr -> nxt
        track = warp_mask(track, flow)
        guide01 = (track[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)

        prob = _predict_prob(model, _read_frame_tensor(curr), device)
        bin01 = (prob >= float(thr)).astype(np.uint8)
        sel01 = _select_cc_by_iou(bin01, guide01, min_area=min_area)

        if iou_track_min > 0:
            iou_val = iou_binary(
                torch.from_numpy(sel01.astype(np.float32))[None, None],
                torch.from_numpy(guide01.astype(np.float32))[None, None],
            )
            if iou_val < float(iou_track_min):
                sel01[:] = 0

        _write_mask(masks_out_dir / curr.name, sel01)

        if sel01.sum() > 0:
            track = torch.from_numpy(sel01.astype(np.float32))[None, None].to(device)


def run(
    images_root: str,
    output_root: str,
    resize: int = 512,
    seed_name: Optional[str] = None,
    backend: str = "baseline",
    teacher_ckpt: str = "",
    thr: float = 0.5,
    min_area: int = 50,
    iou_track_min: float = 0.0,
) -> None:
    images_root_p = Path(images_root)
    output_root_p = Path(output_root)

    backend = (backend or "baseline").strip().lower()

    if backend == "modeltrack":
        if not seed_name:
            raise RuntimeError("backend=modeltrack requires --seed-name")
        if not teacher_ckpt:
            raise RuntimeError("backend=modeltrack requires --teacher-ckpt")
        teacher_ckpt_p = Path(teacher_ckpt)
        if not teacher_ckpt_p.exists():
            raise RuntimeError(f"teacher-ckpt not found: {teacher_ckpt_p}")
    else:
        teacher_ckpt_p = Path(teacher_ckpt) if teacher_ckpt else Path()

    for split in sorted(images_root_p.iterdir()):
        if not split.is_dir():
            continue
        for clip in sorted(split.iterdir()):
            if not clip.is_dir():
                continue
            frames_dir = clip / "frames"
            masks_dir = clip / "masks"
            if not frames_dir.exists() or not masks_dir.exists():
                continue

            out_frames = output_root_p / split.name / clip.name / "frames"
            out_masks = output_root_p / split.name / clip.name / "masks"
            out_flows = output_root_p / "flows" / split.name / clip.name
            ensure_dir(out_frames)
            ensure_dir(out_masks)
            ensure_dir(out_flows)

            if backend == "baseline":
                propagate_baseline.propagate_clip(
                    frames_dir=frames_dir,
                    masks_dir=masks_dir,
                    out_frames_dir=out_frames,
                    out_masks_dir=out_masks,
                    out_flow_dir=out_flows,
                    resize=resize,
                    flows_only=False,
                    seed_name=seed_name,
                )
                continue

            if backend == "modeltrack":
                # 1) Generate resized frames + flows
                propagate_baseline.propagate_clip(
                    frames_dir=frames_dir,
                    masks_dir=masks_dir,
                    out_frames_dir=out_frames,
                    out_masks_dir=out_masks,
                    out_flow_dir=out_flows,
                    resize=resize,
                    flows_only=True,
                    seed_name=seed_name,
                )
                # 2) Remove any stale masks
                for p in out_masks.glob("*.png"):
                    p.unlink(missing_ok=True)
                # 3) Produce tracked masks using model predictions + flow guidance
                _modeltrack_clip(
                    frames_out_dir=out_frames,
                    masks_in_dir=masks_dir,
                    masks_out_dir=out_masks,
                    flow_dir=out_flows,
                    seed_name=seed_name,
                    resize=resize,
                    teacher_ckpt=teacher_ckpt_p,
                    thr=thr,
                    min_area=min_area,
                    iou_track_min=iou_track_min,
                )
                continue

            raise RuntimeError(f"Unknown backend: {backend}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--seed-name", type=str, default=None, help="e.g. 0280.png")
    ap.add_argument("--backend", type=str, default="baseline", choices=["baseline", "modeltrack"])
    ap.add_argument("--teacher-ckpt", type=str, default="")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--iou-track-min", type=float, default=0.0)
    args = ap.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
