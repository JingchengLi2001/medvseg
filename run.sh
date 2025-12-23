#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# MedVSeg one-click to video
# -----------------------------
# 需要的输入（默认值都能改）：
#   DATASET_ROOT : 数据根目录（包含 split/clip_0001/frames,masks）
#   SPLIT        : split 名称（默认 test_easy_seen）
#   CLIP         : clip 名称（默认 clip_0001）
#   SEED_NAME    : 种子帧文件名（默认自动从 masks 里取第一个 png）
#   VIDEO_SRC    : 原始 mp4 路径（用于最终生成叠加视频；不填则用 frames 目录生成）
#   RESIZE       : 传播/训练推理统一分辨率（默认 512）
#
# 一键示例：
#   chmod +x run.sh
#   DATASET_ROOT=data/MED SPLIT=test_easy_seen CLIP=clip_0001 SEED_NAME=0280.png VIDEO_SRC=data/MED/raw/myvideo.mp4 RESIZE=512 ./run.sh

DATASET_ROOT="${DATASET_ROOT:-data/MED}"
SPLIT="${SPLIT:-test_easy_seen}"
CLIP="${CLIP:-clip_0001}"
RESIZE="${RESIZE:-512}"
IOU_TH="${IOU_TH:-0.4}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-3}"
SMOOTH="${SMOOTH:-0.5}"
HALF="${HALF:-1}"
OUT_VIDEO="${OUT_VIDEO:-outputs/seg_overlay.mp4}"
CLEAN="${CLEAN:-1}"

here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"
export PYTHONPATH="$here:${PYTHONPATH:-}"

# 关闭 Albumentations 联网检查（避免反复报错刷屏）
export NO_ALBUMENTATIONS_UPDATE=1
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

info() { printf "\033[36m[INFO]\033[0m %s\n" "$*"; }
warn() { printf "\033[33m[WARN]\033[0m %s\n" "$*"; }
die()  { printf "\033[31m[ERR ]\033[0m %s\n" "$*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "缺少命令：$1"; }

need_cmd python

# -----------------------------
# 0) 基本信息
# -----------------------------
python - <<'PY'
import torch
print(f"Torch {torch.__version__} | CUDA? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    try: print("CUDA device:", torch.cuda.get_device_name(0))
    except Exception: pass
PY

# -----------------------------
# 1) 保证关键依赖存在（不动 torch）
# -----------------------------
info "Ensuring python deps (no torch changes)..."
python - <<'PY'
import importlib, subprocess, sys

def has(mod):
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False

need = []
# 你的模型需要 smp
if not has("segmentation_models_pytorch"):
    need += ["segmentation-models-pytorch==0.3.3", "timm==0.9.2", "efficientnet-pytorch==0.7.1", "pretrainedmodels==0.7.4"]
# 进度条/可视化打印
if not has("tqdm"): need += ["tqdm"]
if not has("yaml"): need += ["pyyaml"]
# OpenCV/albumentations 你已有就不强行重装（避免再引入 numpy ABI 冲突）
if not has("cv2"): need += ["opencv-python-headless==4.10.0.84"]
if not has("albumentations"): need += ["albumentations==1.4.13", "albucore==0.0.16"]

if need:
    print("[INFO] pip install:", " ".join(need))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *need])
else:
    print("[INFO] deps ok.")
PY

python - <<'PY'
import cv2
print("OpenCV:", cv2.__version__)
PY

# -----------------------------
# 1.5) 避免下载 encoder 预训练权重（离线环境很关键）
# -----------------------------
if grep -q "encoder_weights='imagenet'" medvseg/models/student_unet.py 2>/dev/null; then
  sed -i "s/encoder_weights='imagenet'/encoder_weights=None/" medvseg/models/student_unet.py
  info "Patched student_unet.py → encoder_weights=None"
fi

# -----------------------------
# 1.6) 如果没有 stream_realtime.py，就自动写入（保证一键有视频输出）
# -----------------------------
if [ ! -f medvseg/engines/stream_realtime.py ]; then
  warn "medvseg/engines/stream_realtime.py missing → creating it"
  cat > medvseg/engines/stream_realtime.py <<'PY'
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from medvseg.models.student_unet import StudentUNet

def _ensure_writer(save_path: Path, w: int, h: int, fps: float):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(save_path), fourcc, fps, (w, h))
    return writer

def main(model: str, src: str, size: int = 512, half: int = 1, smooth: float = 0.5, save: str = "outputs/seg_overlay.mp4"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = bool(half) and (device == "cuda")
    smooth = max(0.0, min(1.0, float(smooth)))

    net = StudentUNet("resnet34", 3, 1)
    state = torch.load(model, map_location=device)
    net.load_state_dict(state["model"])
    net.to(device).eval()

    src_path = Path(src)
    save_path = Path(save)

    use_video = src_path.is_file() and src_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]
    use_dir = src_path.is_dir()

    if not (use_video or use_dir):
        raise RuntimeError(f"--src must be a video file or a frames directory, got: {src_path}")

    prev_prob = None
    writer = None
    out_frames_dir = None
    fps = 25.0

    if use_dir:
        frames = sorted(src_path.glob("*.png"))
        if not frames:
            raise RuntimeError(f"No png frames under: {src_path}")
        first = cv2.imread(str(frames[0]))
        if first is None:
            raise RuntimeError(f"Cannot read: {frames[0]}")
        H0, W0 = first.shape[:2]
        writer = _ensure_writer(save_path, W0, H0, fps)
        if not writer.isOpened():
            out_frames_dir = save_path.with_suffix("")
            out_frames_dir.mkdir(parents=True, exist_ok=True)

        def frame_iter():
            for p in frames:
                fr = cv2.imread(str(p))
                if fr is None: 
                    continue
                yield p.name, fr
    else:
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {src_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-3:
            fps = 25.0
        ok, first = cap.read()
        cap.release()
        if not ok or first is None:
            raise RuntimeError(f"Cannot read first frame: {src_path}")
        H0, W0 = first.shape[:2]

        writer = _ensure_writer(save_path, W0, H0, fps)
        if not writer.isOpened():
            out_frames_dir = save_path.with_suffix("")
            out_frames_dir.mkdir(parents=True, exist_ok=True)

        def frame_iter():
            cap2 = cv2.VideoCapture(str(src_path))
            idx = 0
            while True:
                ok, fr = cap2.read()
                if not ok:
                    break
                idx += 1
                yield f"{idx:04d}.png", fr
            cap2.release()

    with torch.no_grad():
        for name, frame_bgr in frame_iter():
            H, W = frame_bgr.shape[:2]
            inp = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
            inp_rgb = inp[:, :, ::-1].copy()
            x = torch.from_numpy(inp_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            x = x.to(device)
            if half:
                x = x.half()

            with torch.autocast(device_type="cuda", enabled=half):
                logits = net(x)
                prob = torch.sigmoid(logits)[0, 0].float().cpu().numpy()

            if prev_prob is None:
                prob_s = prob
            else:
                prob_s = smooth * prev_prob + (1.0 - smooth) * prob
            prev_prob = prob_s

            prob_up = cv2.resize(prob_s, (W, H), interpolation=cv2.INTER_LINEAR)
            mask = (prob_up >= 0.5).astype(np.uint8)

            overlay = frame_bgr.copy()
            overlay[mask > 0] = (0, 0, 255)
            out = cv2.addWeighted(frame_bgr, 0.65, overlay, 0.35, 0)

            if writer is not None and writer.isOpened():
                writer.write(out)
            else:
                cv2.imwrite(str(out_frames_dir / name), out)

    if writer is not None and writer.isOpened():
        writer.release()

    if out_frames_dir is not None:
        print("[WARN] VideoWriter failed. Saved overlay frames to:", out_frames_dir)
    else:
        print("[OK] Saved video ->", save_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--save", default="outputs/seg_overlay.mp4")
    args = ap.parse_args()
    main(**vars(args))
PY
fi

# -----------------------------
# 2) 自动确定 SEED_NAME（如果没给）
# -----------------------------
seed_dir="$DATASET_ROOT/$SPLIT/$CLIP/masks"
frame_dir="$DATASET_ROOT/$SPLIT/$CLIP/frames"

[ -d "$frame_dir" ] || die "frames 目录不存在：$frame_dir"
[ -d "$seed_dir" ]  || die "masks 目录不存在：$seed_dir"

if [ -z "${SEED_NAME:-}" ]; then
  SEED_NAME="$(ls -1 "$seed_dir"/*.png 2>/dev/null | head -n1 | xargs -n1 basename || true)"
  [ -n "$SEED_NAME" ] || die "masks 下没有任何 png：$seed_dir"
  info "Auto picked SEED_NAME=$SEED_NAME"
fi

seed_frame="$frame_dir/$SEED_NAME"
seed_mask="$seed_dir/$SEED_NAME"
[ -f "$seed_frame" ] || die "找不到同名帧：$seed_frame（请保证 masks 文件名与 frames 完全一致）"
[ -f "$seed_mask" ]  || die "找不到种子 mask：$seed_mask"

# -----------------------------
# 3) 自动修正种子 mask：尺寸对齐 + 二值化 0/255
# -----------------------------
info "Fixing seed mask (resize to frame, binarize to 0/255)..."
python - <<PY
import cv2, numpy as np
from pathlib import Path

f = Path("$seed_frame")
m = Path("$seed_mask")
img = cv2.imread(str(f), cv2.IMREAD_COLOR)
msk = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
assert img is not None, f"读不到 frame: {f}"
assert msk is not None, f"读不到 mask:  {m}"

H,W = img.shape[:2]
if msk.shape[:2] != (H,W):
    msk = cv2.resize(msk, (W,H), interpolation=cv2.INTER_NEAREST)

msk_bin = ((msk > 0).astype(np.uint8) * 255)
cv2.imwrite(str(m), msk_bin)

print("[OK] seed frame:", img.shape, "| seed mask:", msk_bin.shape,
      "| nonzero:", int(np.count_nonzero(msk_bin)))
PY

# -----------------------------
# 4) 清理旧产物
# -----------------------------
if [ "$CLEAN" = "1" ]; then
  info "Cleaning old outputs..."
  rm -rf outputs/xmem_raw outputs/pseudolabels_clean outputs/runs/unet_r34 outputs/seg_overlay.mp4
fi

# -----------------------------
# 5) 传播 + flows
# -----------------------------
info "Propagate masks (Farneback baseline) with seed=$SEED_NAME ..."
python -m medvseg.engines.propagate_baseline \
  --images-root "$DATASET_ROOT" \
  --output-root outputs/xmem_raw \
  --resize "$RESIZE" \
  --seed-name "$SEED_NAME"

info "Compute flows only ..."
python -m medvseg.engines.propagate_baseline \
  --images-root "$DATASET_ROOT" \
  --output-root outputs/xmem_raw \
  --resize "$RESIZE" \
  --flows-only 1

# -----------------------------
# 6) 同步 frames 到 xmem_raw，并 resize 到 RESIZE（避免尺寸不一致）
# -----------------------------
info "Sync+resize frames into outputs/xmem_raw/<split>/<clip>/frames ..."
python - <<PY
from pathlib import Path
import cv2

data_root = Path("$DATASET_ROOT")
out_root  = Path("outputs/xmem_raw")
resize = int("$RESIZE")

for split in sorted([d for d in data_root.iterdir() if d.is_dir()]):
    for clip in sorted([d for d in split.iterdir() if d.is_dir()]):
        fdir = clip/"frames"
        if not fdir.exists(): 
            continue
        dst = out_root/split.name/clip.name/"frames"
        dst.mkdir(parents=True, exist_ok=True)
        for p in sorted(fdir.glob("*.png")):
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                continue
            if resize > 0:
                im = cv2.resize(im, (resize, resize), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(dst/p.name), im)
print("[OK] frames synced.")
PY

# -----------------------------
# 7) 筛选伪标签
# -----------------------------
info "Filter pseudo labels (IoU_th=$IOU_TH) ..."
python -m medvseg.engines.filter_pseudolabels \
  --pred-root outputs/xmem_raw \
  --flow-root outputs/xmem_raw/flows \
  --output-root outputs/pseudolabels_clean \
  --iou-th "$IOU_TH"

# -----------------------------
# 8) 训练 + 评测
# -----------------------------
train_root="outputs/pseudolabels_clean/$SPLIT"
[ -d "$train_root" ] || die "训练目录不存在：$train_root（检查 filter 输出）"

info "Training student model..."
python -m medvseg.engines.train_student \
  --data-root "$train_root" \
  --val-root  "$train_root" \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR" \
  --save-dir outputs/runs/unet_r34

info "Evaluating..."
python -m medvseg.engines.evaluate \
  --model outputs/runs/unet_r34/best.ckpt \
  --images-root "$train_root" \
  --save-dir outputs/runs/unet_r34/eval

# -----------------------------
# 9) 生成叠加分割视频（优先用 VIDEO_SRC；没有就用 frames 目录）
# -----------------------------
src="${VIDEO_SRC:-}"
if [ -z "$src" ]; then
  warn "VIDEO_SRC not set. Using frames dir as src: $frame_dir"
  src="$frame_dir"
else
  [ -f "$src" ] || die "VIDEO_SRC 不存在：$src"
fi

info "Export overlay video -> $OUT_VIDEO"
python -m medvseg.engines.stream_realtime \
  --model outputs/runs/unet_r34/best.ckpt \
  --src "$src" \
  --size "$RESIZE" --half "$HALF" --smooth "$SMOOTH" \
  --save "$OUT_VIDEO"

info "DONE."
info "1) overlay video: $OUT_VIDEO"
info "2) per-frame preds: outputs/runs/unet_r34/eval/*_pred.png"
