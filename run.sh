#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------
# MedVSeg one-click (cu118, final stable)
# ---------------------------------------
# 用法：
#   bash run.sh
#   DATASET_ROOT=data/YourSet/test_easy_seen RESIZE=512 bash run.sh
# ---------------------------------------

DATASET_ROOT="${DATASET_ROOT:-data/TOY}"
RESIZE="${RESIZE:-512}"

here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"
export PYTHONPATH="$here:${PYTHONPATH:-}"
# 静音 Albumentations 的联网版本检查（run.sh 内有效；你手动跑命令可在 shell 里也 export 一下）
export NO_ALBUMENTATIONS_UPDATE=1
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1

info() { printf "\033[36m[INFO]\033[0m %s\n" "$*"; }
warn() { printf "\033[33m[WARN]\033[0m %s\n" "$*"; }
die()  { printf "\033[31m[ERR ]\033[0m %s\n" "$*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "缺少命令：$1"; }

# 0) Python & Torch
need_cmd python
python - <<'PY'
import torch
print(f"Torch {torch.__version__} | CUDA? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    try: print("CUDA device:", torch.cuda.get_device_name(0))
    except Exception: pass
PY

# 1) 固定与 OpenCV 4.10 兼容的一组科学包（不动 torch 系列）
python - <<'PY'
import importlib, subprocess, sys
want = {
  "numpy": "1.26.4",
  "pandas": "2.0.3",
  "scipy": "1.11.4",
  "scikit-learn": "1.3.2",
  "albucore": "0.0.16",
  "albumentations": "1.4.13",
  "opencv-python-headless": "4.10.0.84",
}
need = []
for pkg, ver in want.items():
    try:
        m = importlib.import_module(pkg.replace("-", "_"))
        cur = getattr(m, "__version__", None)
        if cur != ver: need.append(f"{pkg}=={ver}")
    except Exception:
        need.append(f"{pkg}=={ver}")
if need:
    print("[INFO] Installing/pinning:", " ".join(need))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--force-reinstall", *need])
else:
    print("[INFO] All pinned deps already satisfied.")
PY

# OpenCV 健康检查
python - <<'PY'
import cv2; print("OpenCV:", cv2.__version__)
PY

# 1.5) 避免下载预训练权重：把 SMP 的 encoder_weights='imagenet' 改为 None（若存在）
if grep -q "encoder_weights='imagenet'" medvseg/models/student_unet.py 2>/dev/null; then
  sed -i "s/encoder_weights='imagenet'/encoder_weights=None/" medvseg/models/student_unet.py
  info "Patched student_unet.py → encoder_weights=None"
fi

# 2) 生成 toy（仅默认 data/TOY 时）
if [ "$DATASET_ROOT" = "data/TOY" ]; then
  info "Generating toy dataset..."
  python -m medvseg.engines.make_toyset --root "$DATASET_ROOT"
fi

# 3) 传播（优先 XMem，否则 Farneback）
XMEM_WEIGHT="$here/third_party/XMem/saves/xmem.pth"
if [ -f "$XMEM_WEIGHT" ]; then
  info "Using XMem propagation"
  python -m medvseg.engines.propagate_xmem \
    --images-root "$DATASET_ROOT" \
    --output-root outputs/xmem_raw \
    --xmem-root third_party/XMem
else
  warn "XMem weight not found → using Farneback baseline"
  python -m medvseg.engines.propagate_baseline \
    --images-root "$DATASET_ROOT" \
    --output-root outputs/xmem_raw \
    --resize "$RESIZE"
fi

# 4) 光流（写入 outputs/xmem_raw/flows）
info "Computing flows (Farneback)"
python -m medvseg.engines.propagate_baseline \
  --images-root "$DATASET_ROOT" \
  --output-root outputs/xmem_raw \
  --resize "$RESIZE" \
  --flows-only 1

# 4.5) 为筛选补齐帧：把原始 frames 同步到 xmem_raw/<split>/<clip>/frames
info "Sync frames into xmem_raw for filtering"
for split_dir in "$DATASET_ROOT"/*; do
  [ -d "$split_dir" ] || continue
  split="$(basename "$split_dir")"
  for clip_dir in "$split_dir"/*; do
    [ -d "$clip_dir" ] || continue
    clip="$(basename "$clip_dir")"
    src_frames="$clip_dir/frames"
    dst_frames="outputs/xmem_raw/$split/$clip/frames"
    if [ -d "$src_frames" ]; then
      mkdir -p "$dst_frames"
      cp -n "$src_frames"/*.png "$dst_frames/" 2>/dev/null || true
    fi
  done
done

# 5) 一致性筛选（先尝试）
info "Filtering pseudo labels"
python -m medvseg.engines.filter_pseudolabels \
  --pred-root  outputs/xmem_raw \
  --flow-root  outputs/xmem_raw/flows \
  --output-root outputs/pseudolabels_clean \
  --iou-th 0.4 || true

# 5.1) 若无有效配对 → 强制兜底构建（frames ∩ masks 的交集）
has_pairs=$(python - <<'PY'
from pathlib import Path
root=Path("outputs/pseudolabels_clean")
pairs=0
if root.exists():
    for clip in root.rglob("frames"):
        mdir=clip.parent/"masks"
        if not mdir.exists(): continue
        for f in clip.glob("*.png"):
            if (mdir/f.name).exists():
                print("1"); raise SystemExit(0)
print("0")
PY
)
if [ "$has_pairs" != "1" ]; then
  warn "No valid pairs after filtering. Building dataset from xmem_raw (frames ∩ masks)..."
  rm -rf outputs/pseudolabels_clean
  python - <<'PY'
from pathlib import Path, shutil
root_out = Path("outputs/pseudolabels_clean")
root_x   = Path("outputs/xmem_raw")
for split in sorted([d for d in root_x.iterdir() if d.is_dir() and d.name!="flows"]):
    for clip in sorted([d for d in split.iterdir() if d.is_dir()]):
        fdir = clip/"frames"; mdir = clip/"masks"
        if not fdir.exists() or not mdir.exists(): continue
        frames = {p.name for p in fdir.glob("*.png")}
        masks  = {p.name for p in mdir.glob("*.png")}
        inter  = sorted(frames & masks)
        if not inter:
            print("Skip (no intersection)", clip); continue
        out = root_out/split.name/clip.name
        (out/"frames").mkdir(parents=True, exist_ok=True)
        (out/"masks").mkdir(parents=True, exist_ok=True)
        for name in inter:
            shutil.copy2(str(fdir/name), str(out/"frames"/name))
            shutil.copy2(str(mdir/name), str(out/"masks"/name))
        print("Built", out, "pairs=", len(inter))
PY
fi

# 6) 选择正确的数据根（FrameMaskDataset 不包含 split 层）
DATA_ROOT_BASE="outputs/pseudolabels_clean"
if compgen -G "$DATA_ROOT_BASE/clip_*/frames/*.png" >/dev/null; then
  DATA_ROOT_TRAIN="$DATA_ROOT_BASE"
elif compgen -G "$DATA_ROOT_BASE"/*/clip_*/frames/*.png >/dev/null; then
  # 只退一层，指向 split 目录（例如 test_easy_seen）
  DATA_ROOT_TRAIN="$(dirname "$(ls -d "$DATA_ROOT_BASE"/*/clip_* | head -n1)")"
else
  die "找不到可用的数据目录：$DATA_ROOT_BASE（或其子目录）下缺少 clip_*/frames/*.png"
fi
echo "[INFO] Using dataset root: $DATA_ROOT_TRAIN"

# 7) 训练
info "Training student model"
python -m medvseg.engines.train_student \
  --data-root "$DATA_ROOT_TRAIN" \
  --val-root  "$DATA_ROOT_TRAIN" \
  --epochs 5 --batch-size 4 --lr 1e-3 \
  --save-dir outputs/runs/unet_r34

# 8) 评测
info "Evaluating"
python -m medvseg.engines.evaluate \
  --model outputs/runs/unet_r34/best.ckpt \
  --images-root "$DATA_ROOT_TRAIN" \
  --save-dir outputs/runs/unet_r34/eval

info "All done. See outputs/runs/unet_r34/eval"
