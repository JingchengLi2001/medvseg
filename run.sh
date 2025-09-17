#!/usr/bin/env bash
set -euo pipefail

# -------------------------------
# MedVSeg one-click (macOS/Linux)
# -------------------------------
# 用法：
#   bash run.sh                          # 用 toy 数据跑通
#   DATASET_ROOT=data/YourSet/test_easy_seen bash run.sh  # 换你的数据
# 环境策略：
#   - 优先使用 conda env: medvseg；否则使用 venv
#   - macOS 强制 CPU 版 PyTorch；Ubuntu 若检测到 nvidia-smi 则装 cu121 版，否则 CPU 版
# -------------------------------

DATASET_ROOT="${DATASET_ROOT:-data/TOY}"
RESIZE="${RESIZE:-512}"
ENV_NAME="${ENV_NAME:-medvseg}"

here="$(cd "$(dirname "$0")" && pwd)"

info() { printf "\033[36m[INFO]\033[0m %s\n" "$*"; }
warn() { printf "\033[33m[WARN]\033[0m %s\n" "$*"; }
die()  { printf "\033[31m[ERR ]\033[0m %s\n" "$*" >&2; exit 1; }

# 0) 选择 conda 或 venv
PY="python3"
if command -v conda >/dev/null 2>&1; then
  info "Detected conda. Using env: $ENV_NAME"
  conda create -n "$ENV_NAME" python=3.10 -y >/dev/null 2>&1 || true
  PY="conda run -n $ENV_NAME python"
else
  info "Conda not found. Using venv: .venv"
  [ -d .venv ] || python3 -m venv .venv
  . .venv/bin/activate
  PY="python"
fi

# 1) 安装 PyTorch
install_torch() {
  if [[ "$OSTYPE" == "darwin"* ]]; then
    info "macOS detected → installing CPU PyTorch"
    $PY -m pip install --upgrade pip setuptools wheel
    $PY -m pip install "torch>=2.3" "torchvision>=0.18" "torchaudio>=2.3"
  else
    if command -v nvidia-smi >/dev/null 2>&1; then
      info "Linux + NVIDIA GPU → installing cu121 wheels"
      $PY -m pip install --upgrade pip setuptools wheel
      $PY -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
    else
      warn "No GPU detected → installing CPU PyTorch (can run, just slower)"
      $PY -m pip install --upgrade pip setuptools wheel
      $PY -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
    fi
  fi
  $PY - <<'PY'
import torch
print("Torch", torch.__version__, "CUDA?", torch.cuda.is_available())
PY
}
install_torch

# 2) 安装项目依赖
info "Installing project requirements..."
$PY -m pip install -r requirements.txt || true
# OpenCV 出错时回退到 headless
$PY - <<'PY'
try:
    import cv2
    print("OpenCV version:", cv2.__version__)
except Exception as e:
    import sys
    sys.exit(1)
PY
if [ "$?" -ne 0 ]; then
  warn "opencv-python install failed, trying headless build..."
  $PY -m pip install opencv-python-headless==4.10.0.84
fi

# 3) 生成 toy 数据（仅当默认 data/TOY 时）
if [ "$DATASET_ROOT" = "data/TOY" ]; then
  info "Generating toy dataset..."
  $PY medvseg/engines/make_toyset.py --root "$DATASET_ROOT"
fi

# 4) 传播：优先 XMem（若权重存在），否则 Farneback 基线
XMEM_WEIGHT="$here/third_party/XMem/saves/xmem.pth"
if [ -f "$XMEM_WEIGHT" ]; then
  info "Using XMem propagation"
  $PY medvseg/engines/propagate_xmem.py \
    --images-root "$DATASET_ROOT" \
    --output-root outputs/xmem_raw \
    --xmem-root third_party/XMem
else
  warn "XMem weight not found → using Farneback baseline"
  $PY medvseg/engines/propagate_baseline.py \
    --images-root "$DATASET_ROOT" \
    --output-root outputs/xmem_raw \
    --resize "$RESIZE"
fi

# 5) 产出光流（Farneback，用于一致性筛选）
info "Computing flows (Farneback)"
$PY medvseg/engines/propagate_baseline.py \
  --images-root "$DATASET_ROOT" \
  --output-root outputs/xmem_raw \
  --resize "$RESIZE" \
  --flows-only 1

# 6) 一致性筛选
info "Filtering pseudo labels"
$PY medvseg/engines/filter_pseudolabels.py \
  --pred-root outputs/xmem_raw \
  --flow-root outputs/xmem_raw/flows \
  --output-root outputs/pseudolabels_clean \
  --iou-th 0.7

# 7) 训练
info "Training student model"
$PY medvseg/engines/train_student.py \
  --data-root outputs/pseudolabels_clean \
  --val-root  outputs/pseudolabels_clean \
  --epochs 5 --batch-size 4 --lr 1e-3 \
  --save-dir outputs/runs/unet_r34

# 8) 评测
info "Evaluating"
$PY medvseg/engines/evaluate.py \
  --model outputs/runs/unet_r34/best.ckpt \
  --images-root outputs/pseudolabels_clean \
  --save-dir outputs/runs/unet_r34/eval

info "All done. See outputs/runs/unet_r34/eval"
