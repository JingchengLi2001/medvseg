import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from medvseg.models.student_unet import StudentUNet


def _iter_frames_from_dir(frames_dir: Path):
    frames = sorted(frames_dir.glob("*.png"))
    for p in frames:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        yield p.name, img


def _iter_frames_from_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 25.0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        yield f"{idx:04d}.png", frame, fps
    cap.release()


def _ensure_writer(save_path: Path, w: int, h: int, fps: float):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))
    return writer


def main(model: str, src: str, size: int = 512, half: int = 1, smooth: float = 0.5, save: str = "outputs/realtime.mp4"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = bool(half) and (device == "cuda")
    smooth = float(smooth)
    smooth = max(0.0, min(1.0, smooth))

    # load model
    net = StudentUNet("resnet34", 3, 1)
    state = torch.load(model, map_location=device)
    net.load_state_dict(state["model"])
    net.to(device).eval()

    src_path = Path(src)
    save_path = Path(save)

    # decide source
    use_video = src_path.is_file() and src_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]
    use_dir = src_path.is_dir()

    if not (use_video or use_dir):
        raise RuntimeError(f"--src must be a video file or a frames directory, got: {src_path}")

    prev_prob = None
    writer = None
    out_frames_dir = None
    fps = 25.0

    if use_dir:
        # frames dir: output will be same size as original frames
        gen = _iter_frames_from_dir(src_path)
        first_name, first_frame = next(gen, (None, None))
        if first_frame is None:
            raise RuntimeError(f"No readable png frames under: {src_path}")
        H0, W0 = first_frame.shape[:2]
        writer = _ensure_writer(save_path, W0, H0, fps)
        if not writer.isOpened():
            # fallback: write pngs
            out_frames_dir = save_path.with_suffix("")
            out_frames_dir.mkdir(parents=True, exist_ok=True)
        # re-create generator (include first frame)
        def regen():
            yield first_name, first_frame
            for name, fr in gen:
                yield name, fr
        frame_iter = regen()

    else:
        # video: need fps from capture
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {src_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-3:
            fps = 25.0
        ret, first_frame = cap.read()
        cap.release()
        if not ret or first_frame is None:
            raise RuntimeError(f"Cannot read first frame from video: {src_path}")
        H0, W0 = first_frame.shape[:2]
        writer = _ensure_writer(save_path, W0, H0, fps)
        if not writer.isOpened():
            out_frames_dir = save_path.with_suffix("")
            out_frames_dir.mkdir(parents=True, exist_ok=True)

        def frame_iter_video():
            cap2 = cv2.VideoCapture(str(src_path))
            idx = 0
            while True:
                ok, fr = cap2.read()
                if not ok:
                    break
                idx += 1
                yield f"{idx:04d}.png", fr
            cap2.release()
        frame_iter = frame_iter_video()

    # inference loop
    with torch.no_grad():
        for name, frame_bgr in frame_iter:
            H, W = frame_bgr.shape[:2]

            # network input: resize to (size,size)
            inp = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
            inp_rgb = inp[:, :, ::-1].copy()
            x = torch.from_numpy(inp_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            x = x.to(device)
            if half:
                x = x.half()

            with torch.autocast(device_type="cuda", enabled=half):
                logits = net(x)
                prob = torch.sigmoid(logits)[0, 0].float().detach().cpu().numpy()  # (size,size) in [0,1]

            # temporal smoothing
            if prev_prob is None:
                prob_s = prob
            else:
                prob_s = smooth * prev_prob + (1.0 - smooth) * prob
            prev_prob = prob_s

            # upsample mask back to original resolution
            prob_up = cv2.resize(prob_s, (W, H), interpolation=cv2.INTER_LINEAR)
            mask = (prob_up >= 0.5).astype(np.uint8)

            # overlay
            overlay = frame_bgr.copy()
            overlay[mask > 0] = (0, 0, 255)  # red in BGR
            out = cv2.addWeighted(frame_bgr, 0.65, overlay, 0.35, 0)

            if writer is not None and writer.isOpened():
                writer.write(out)
            else:
                # fallback: save frames
                cv2.imwrite(str(out_frames_dir / name), out)

    if writer is not None and writer.isOpened():
        writer.release()

    if out_frames_dir is not None:
        print("[WARN] VideoWriter failed. Saved overlay frames to:", out_frames_dir)
        print("If you have ffmpeg, encode with:")
        print(f"ffmpeg -hide_banner -y -r {fps} -i {out_frames_dir}/%04d.png -c:v libx264 -pix_fmt yuv420p {save_path}")
    else:
        print("[OK] Saved video ->", save_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", required=True, help="video path (.mp4) or frames directory")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--save", default="outputs/realtime.mp4")
    args = ap.parse_args()
    main(**vars(args))
