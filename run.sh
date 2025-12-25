#!/usr/bin/env bash
set -euo pipefail

# =============================
# MedVSeg One-Click (Single/Multi Seed)
# propagate -> flows -> filter -> merge -> train -> eval -> overlay video
# =============================

DATASET_ROOT="${DATASET_ROOT:-data/MED}"
SPLIT="${SPLIT:-test_easy_seen}"
CLIP="${CLIP:-clip_0001}"
RESIZE="${RESIZE:-512}"

IOU_TH="${IOU_TH:-0.4}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-3}"

HALF="${HALF:-1}"
SMOOTH="${SMOOTH:-0.5}"
PRED_TH="${PRED_TH:-0.5}"          # overlay threshold
MIN_MASK_PIXELS="${MIN_MASK_PIXELS:-50}"  # drop empty/tiny masks

MULTI_SEED="${MULTI_SEED:-0}"      # 1=use multiple seeds
SEED_NAME="${SEED_NAME:-}"         # single seed name (e.g. 0001.png)
SEED_LIST="${SEED_LIST:-}"         # optional: "0001.png 0006.png ..."

VIDEO_SRC="${VIDEO_SRC:-}"         # mp4 path (recommended)
OUT_VIDEO="${OUT_VIDEO:-outputs/seg_overlay.mp4}"
OUT_FPS="${OUT_FPS:-}"             # only used when src is a frames dir
FPS_EXTRACT="${FPS_EXTRACT:-5}"    # your extracted fps (e.g. 5)
CLEAN="${CLEAN:-1}"

PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"
export PYTHONPATH="$here:${PYTHONPATH:-}"

export NO_ALBUMENTATIONS_UPDATE=1
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1
export PIP_INDEX_URL="$PIP_INDEX_URL"

info() { printf "\033[36m[INFO]\033[0m %s\n" "$*"; }
warn() { printf "\033[33m[WARN]\033[0m %s\n" "$*"; }
die()  { printf "\033[31m[ERR ]\033[0m %s\n" "$*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "缺少命令：$1"; }

need_cmd python
mkdir -p outputs/logs

# -----------------------------
# 0) Print env summary
# -----------------------------
info "DATASET_ROOT=$DATASET_ROOT | SPLIT=$SPLIT | CLIP=$CLIP | RESIZE=$RESIZE"
info "MULTI_SEED=$MULTI_SEED | SEED_NAME=${SEED_NAME:-<auto>} | SEED_LIST=${SEED_LIST:-<auto>}"
info "VIDEO_SRC=${VIDEO_SRC:-<empty>} | OUT_VIDEO=$OUT_VIDEO | IOU_TH=$IOU_TH | EPOCHS=$EPOCHS | BS=$BATCH_SIZE"

python - <<'PY'
import torch
print(f"Torch {torch.__version__} | CUDA? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    try: print("CUDA device:", torch.cuda.get_device_name(0))
    except Exception: pass
PY

# -----------------------------
# 1) Ensure python deps (NO torch changes)
# -----------------------------
info "Ensuring python deps (no torch changes)..."
python - <<'PY'
import importlib, subprocess, sys, os

def has(mod):
    try:
        importlib.import_module(mod); return True
    except Exception:
        return False

need = []
# project basics
if not has("yaml"): need += ["pyyaml"]
if not has("tqdm"): need += ["tqdm"]

# cv2/numpy usually exist; don't force reinstall to avoid ABI issues
# training model deps (smp)
if not has("segmentation_models_pytorch"):
    need += ["segmentation-models-pytorch==0.3.3", "timm==0.9.2", "efficientnet-pytorch==0.7.1", "pretrainedmodels==0.7.4"]

# albumentations (if your dataset pipeline uses it)
if not has("albumentations"):
    need += ["albumentations==1.4.13", "albucore==0.0.16"]

if need:
    print("[INFO] pip install:", " ".join(need))
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--no-cache-dir", "--root-user-action=ignore", *need], env=env)
else:
    print("[INFO] deps ok.")
PY

python - <<'PY'
import cv2
print("OpenCV:", cv2.__version__)
PY

# -----------------------------
# 1.5) Offline patch for encoder weights (avoid downloading imagenet)
# -----------------------------
if [ -f medvseg/models/student_unet.py ]; then
  if grep -q "encoder_weights='imagenet'" medvseg/models/student_unet.py 2>/dev/null; then
    sed -i "s/encoder_weights='imagenet'/encoder_weights=None/" medvseg/models/student_unet.py
    info "Patched student_unet.py → encoder_weights=None"
  fi
fi

# -----------------------------
# 2) Resolve seed list
# -----------------------------
seed_dir="$DATASET_ROOT/$SPLIT/$CLIP/masks"
frame_dir="$DATASET_ROOT/$SPLIT/$CLIP/frames"
[ -d "$frame_dir" ] || die "frames 目录不存在：$frame_dir"
[ -d "$seed_dir" ]  || die "masks 目录不存在：$seed_dir"

seeds=()
if [ "$MULTI_SEED" = "1" ]; then
  if [ -n "$SEED_LIST" ]; then
    # user provided
    read -r -a seeds <<< "$SEED_LIST"
  else
    # auto collect
    mapfile -t seeds < <(find "$seed_dir" -maxdepth 1 -type f -name "*.png" -printf "%f\n" | sort -V)
  fi
  [ "${#seeds[@]}" -gt 0 ] || die "MULTI_SEED=1 但 masks 下没有任何 png：$seed_dir"
  info "Multi-seed mode. Seeds: ${seeds[*]}"
else
  if [ -z "$SEED_NAME" ]; then
    SEED_NAME="$(find "$seed_dir" -maxdepth 1 -type f -name "*.png" -printf "%f\n" | sort -V | head -n1 || true)"
    [ -n "$SEED_NAME" ] || die "masks 下没有任何 png：$seed_dir"
    info "Auto picked SEED_NAME=$SEED_NAME"
  fi
  seeds=("$SEED_NAME")
  info "Single-seed mode. Seed: ${seeds[0]}"
fi

# -----------------------------
# 3) Fix seed masks: resize to frame + binarize 0/255
# -----------------------------
info "Fixing seed masks (resize to frame, binarize to 0/255)..."
for s in "${seeds[@]}"; do
  seed_frame="$frame_dir/$s"
  seed_mask="$seed_dir/$s"
  [ -f "$seed_frame" ] || die "找不到同名帧：$seed_frame（masks 文件名必须与 frames 完全一致）"
  [ -f "$seed_mask" ]  || die "找不到种子 mask：$seed_mask"
  python - "$seed_frame" "$seed_mask" <<'PY'
import sys, cv2, numpy as np
f, m = sys.argv[1], sys.argv[2]
img = cv2.imread(f, cv2.IMREAD_COLOR)
msk = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
assert img is not None, f"Cannot read frame: {f}"
assert msk is not None, f"Cannot read mask: {m}"
H,W = img.shape[:2]
if msk.shape[:2] != (H,W):
    msk = cv2.resize(msk, (W,H), interpolation=cv2.INTER_NEAREST)
msk = ((msk > 0).astype(np.uint8) * 255)
cv2.imwrite(m, msk)
print(f"[OK] {m.split('/')[-1]} -> shape={msk.shape} nonzero={(msk>0).sum()}")
PY
done

# -----------------------------
# 4) Clean outputs (never touch your dataset)
# -----------------------------
if [ "$CLEAN" = "1" ]; then
  info "Cleaning old outputs (only outputs/)..."
  rm -rf outputs/xmem_raw_seed_* outputs/pseudolabels_clean_seed_* outputs/pseudolabels_clean_multi outputs/runs/unet_r34 outputs/seg_overlay.mp4 || true
fi

prune_to_one_clip() {
  local ROOT="$1" SPL="$2" CLP="$3"
  [ -d "$ROOT" ] || return 0
  # keep only $SPL and flows
  find "$ROOT" -mindepth 1 -maxdepth 1 -type d ! -name "flows" ! -name "$SPL" -exec rm -rf {} + 2>/dev/null || true
  # keep only $CLP under split
  if [ -d "$ROOT/$SPL" ]; then
    find "$ROOT/$SPL" -mindepth 1 -maxdepth 1 -type d ! -name "$CLP" -exec rm -rf {} + 2>/dev/null || true
  fi
  # prune flows similarly
  if [ -d "$ROOT/flows" ]; then
    find "$ROOT/flows" -mindepth 1 -maxdepth 1 -type d ! -name "$SPL" -exec rm -rf {} + 2>/dev/null || true
    if [ -d "$ROOT/flows/$SPL" ]; then
      find "$ROOT/flows/$SPL" -mindepth 1 -maxdepth 1 -type d ! -name "$CLP" -exec rm -rf {} + 2>/dev/null || true
    fi
  fi
}

sync_resize_frames_into() {
  local DST_FRAMES="$1"
  mkdir -p "$DST_FRAMES"
  python - "$frame_dir" "$DST_FRAMES" "$RESIZE" <<'PY'
import sys, cv2
from pathlib import Path
src = Path(sys.argv[1]); dst = Path(sys.argv[2]); size = int(sys.argv[3])
dst.mkdir(parents=True, exist_ok=True)
for p in sorted(src.glob("*.png")):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None: 
        continue
    if size > 0:
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(dst / p.name), im)
print("[OK] synced frames ->", dst)
PY
}

drop_empty_masks_inplace() {
  local ROOT="$1"
  python - "$ROOT" "$SPLIT" "$CLIP" "$MIN_MASK_PIXELS" <<'PY'
import sys, cv2
from pathlib import Path
root = Path(sys.argv[1]); split=sys.argv[2]; clip=sys.argv[3]; minpix=int(sys.argv[4])
mdir = root/split/clip/"masks"
fdir = root/split/clip/"frames"
if not mdir.exists():
    print("[WARN] no masks dir:", mdir); sys.exit(0)

kept=0; removed=0
for mp in sorted(mdir.glob("*.png")):
    m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    if m is None:
        removed += 1
        mp.unlink(missing_ok=True)
        continue
    nz = int((m>0).sum())
    if nz < minpix:
        removed += 1
        mp.unlink(missing_ok=True)
        fp = fdir/mp.name
        if fp.exists():
            fp.unlink(missing_ok=True)
    else:
        kept += 1

print(f"[OK] drop-empty: kept={kept} removed={removed} (minpix={minpix})")
PY
}

# ---- restrict propagate to ONLY current split/clip (avoid other clips missing this seed) ----
TMP_IMAGES_ROOT="outputs/_tmp_images_root_${SPLIT}_${CLIP}"
rm -rf "$TMP_IMAGES_ROOT"
mkdir -p "$TMP_IMAGES_ROOT/$SPLIT"
ln -s "$(realpath "$DATASET_ROOT/$SPLIT/$CLIP")" "$TMP_IMAGES_ROOT/$SPLIT/$CLIP"

# -----------------------------
# 5) Per-seed: propagate + flows + sync frames + filter + drop empty
# -----------------------------
clean_roots=()
for s in "${seeds[@]}"; do
  tag="${s%.png}"
  out_raw="outputs/xmem_raw_seed_${tag}"
  out_clean="outputs/pseudolabels_clean_seed_${tag}"
  clean_roots+=("$out_clean")

  info "===== Seed $s (tag=$tag) ====="
  rm -rf "$out_raw" "$out_clean" || true

  info "Propagate -> $out_raw"
  python -m medvseg.engines.propagate_baseline \
    --images-root "$TMP_IMAGES_ROOT" \
    --output-root "$out_raw" \
    --resize "$RESIZE" \
    --seed-name "$s" 2>&1 | tee "outputs/logs/propagate_seed_${tag}.log"

  prune_to_one_clip "$out_raw" "$SPLIT" "$CLIP"

  info "Compute flows -> $out_raw/flows"
  python -m medvseg.engines.propagate_baseline \
    --images-root "$TMP_IMAGES_ROOT" \
    --output-root "$out_raw" \
    --resize "$RESIZE" \
    --flows-only 1 2>&1 | tee "outputs/logs/flows_seed_${tag}.log"

  prune_to_one_clip "$out_raw" "$SPLIT" "$CLIP"

  # ensure frames exist under pred_root (and resized), consistent with your old run.sh
  sync_resize_frames_into "$out_raw/$SPLIT/$CLIP/frames"

  info "Filter pseudo labels (IoU_th=$IOU_TH) -> $out_clean"
  python -m medvseg.engines.filter_pseudolabels \
    --pred-root "$out_raw" \
    --flow-root "$out_raw/flows" \
    --output-root "$out_clean" \
    --iou-th "$IOU_TH" 2>&1 | tee "outputs/logs/filter_seed_${tag}.log"

  # drop empty/tiny masks to avoid training all-background
  drop_empty_masks_inplace "$out_clean"

  kept="$(ls -1 "$out_clean/$SPLIT/$CLIP/masks"/*.png 2>/dev/null | wc -l || true)"
  info "Seed $s final kept masks = $kept"
done

# -----------------------------
# 6) Merge seeds (union masks per frame) -> outputs/pseudolabels_clean_multi/<split>/<clip>
# -----------------------------
out_multi="outputs/pseudolabels_clean_multi"
info "Merging seeds into -> $out_multi/$SPLIT/$CLIP (union masks) ..."
python - "$out_multi" "$SPLIT" "$CLIP" "$RESIZE" "$frame_dir" "${clean_roots[@]}" <<'PY'
import sys, cv2, numpy as np
from pathlib import Path

out_multi = Path(sys.argv[1])
split, clip = sys.argv[2], sys.argv[3]
size = int(sys.argv[4])
frame_dir = Path(sys.argv[5])
roots = [Path(p) for p in sys.argv[6:]]

dst_clip = out_multi/split/clip
dst_f = dst_clip/"frames"
dst_m = dst_clip/"masks"
dst_f.mkdir(parents=True, exist_ok=True)
dst_m.mkdir(parents=True, exist_ok=True)

# collect all mask names
names = set()
for r in roots:
    mdir = r/split/clip/"masks"
    if not mdir.exists(): 
        continue
    for p in mdir.glob("*.png"):
        names.add(p.name)
names = sorted(names)

if not names:
    raise SystemExit(f"[ERR] No masks to merge. Check filter outputs under: {roots}")

kept = 0
for name in names:
    merged = None
    for r in roots:
        mp = r/split/clip/"masks"/name
        if not mp.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None: 
            continue
        if (m.shape[0] != size) or (m.shape[1] != size):
            m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
        m = (m > 0).astype(np.uint8) * 255
        merged = m if merged is None else np.maximum(merged, m)
    if merged is None:
        continue

    # write mask
    cv2.imwrite(str(dst_m/name), merged)

    # write corresponding frame (resized to size)
    fp = frame_dir/name
    if not fp.exists():
        # allow missing frame; but usually should exist
        continue
    im = cv2.imread(str(fp), cv2.IMREAD_COLOR)
    if im is None:
        continue
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(dst_f/name), im)
    kept += 1

print(f"[OK] merged masks/frames kept={kept} -> {dst_clip}")
PY

train_root="$out_multi/$SPLIT"
[ -d "$train_root/$CLIP" ] || die "合并后的训练目录不存在：$train_root/$CLIP"

# -----------------------------
# 7) Train + Eval
# -----------------------------
info "Training student model..."
python -m medvseg.engines.train_student \
  --data-root "$train_root" \
  --val-root  "$train_root" \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR" \
  --save-dir outputs/runs/unet_r34 2>&1 | tee outputs/logs/train.log

info "Evaluating..."
python -m medvseg.engines.evaluate \
  --model outputs/runs/unet_r34/best.ckpt \
  --images-root "$train_root" \
  --save-dir outputs/runs/unet_r34/eval 2>&1 | tee outputs/logs/eval.log

# -----------------------------
# 8) Extra: evaluate on your MANUAL seeds only (meaningful metrics)
# -----------------------------
info "Evaluating on MANUAL seed frames only (real labeled frames)..."
python - "$frame_dir" "$seed_dir" outputs/runs/unet_r34/best.ckpt "$RESIZE" "$PRED_TH" "${seeds[@]}" <<'PY'
import sys, cv2, numpy as np, torch
from pathlib import Path
from medvseg.models.student_unet import StudentUNet

frame_dir = Path(sys.argv[1]); mask_dir = Path(sys.argv[2])
ckpt = sys.argv[3]; size=int(sys.argv[4]); th=float(sys.argv[5])
seed_names = sys.argv[6:]

device = "cuda" if torch.cuda.is_available() else "cpu"
net = StudentUNet("resnet34", 3, 1).to(device).eval()
state = torch.load(ckpt, map_location=device)
sd = state["model"] if isinstance(state, dict) and "model" in state else state
net.load_state_dict(sd)

def dice_iou(pred, gt):
    pred = (pred>0).astype(np.uint8)
    gt = (gt>0).astype(np.uint8)
    inter = (pred & gt).sum()
    a = pred.sum(); b = gt.sum()
    dice = (2*inter) / (a+b+1e-6)
    iou = inter / (a+b-inter+1e-6)
    return float(dice), float(iou), int(a), int(b)

ds=[]; is_=[]
for name in seed_names:
    fp = frame_dir/name
    mp = mask_dir/name
    im = cv2.imread(str(fp), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    if im is None or gt is None:
        continue
    H,W = im.shape[:2]
    x = cv2.resize(im, (size,size), interpolation=cv2.INTER_LINEAR)[:,:,::-1].copy()
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).float()/255.0
    x = x.to(device)
    with torch.no_grad():
        logits = net(x)
        prob = torch.sigmoid(logits)[0,0].detach().cpu().numpy()
    pred = (prob >= th).astype(np.uint8)*255
    pred_up = cv2.resize(pred, (W,H), interpolation=cv2.INTER_NEAREST)
    d,i,a,b = dice_iou(pred_up, gt)
    ds.append(d); is_.append(i)
    print(f"[SEED] {name} dice={d:.4f} iou={i:.4f} pred_fg={a} gt_fg={b}")

if ds:
    print(f"[SEED-MEAN] dice={np.mean(ds):.4f} iou={np.mean(is_):.4f} (th={th})")
else:
    print("[WARN] no seed metrics computed.")
PY

# -----------------------------
# 9) Overlay video (prefer VIDEO_SRC to keep real duration/fps)
# -----------------------------
src="$VIDEO_SRC"
if [ -n "$src" ]; then
  [ -f "$src" ] || die "VIDEO_SRC 不存在：$src"
  info "Overlay source = VIDEO_SRC ($src)  -> keep real fps/duration"
else
  src="$frame_dir"
  # if frames mode, set fps
  if [ -z "$OUT_FPS" ]; then OUT_FPS="$FPS_EXTRACT"; fi
  warn "VIDEO_SRC not set. Using frames dir as src: $src (OUT_FPS=$OUT_FPS, duration depends on it)"
fi

info "Export overlay video -> $OUT_VIDEO"
python - "$src" outputs/runs/unet_r34/best.ckpt "$RESIZE" "$HALF" "$SMOOTH" "$PRED_TH" "$OUT_VIDEO" "${OUT_FPS:-}" <<'PY'
import sys, cv2, numpy as np, torch
from pathlib import Path
from medvseg.models.student_unet import StudentUNet

src = Path(sys.argv[1])
ckpt = sys.argv[2]
size = int(sys.argv[3])
half = int(sys.argv[4]) == 1
smooth = float(sys.argv[5])
th = float(sys.argv[6])
save = Path(sys.argv[7])
fps_arg = sys.argv[8].strip() if len(sys.argv) > 8 else ""

device = "cuda" if torch.cuda.is_available() else "cpu"
half = half and (device=="cuda")

net = StudentUNet("resnet34", 3, 1).to(device).eval()
state = torch.load(ckpt, map_location=device)
sd = state["model"] if isinstance(state, dict) and "model" in state else state
net.load_state_dict(sd)

def ensure_writer(path, w, h, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, float(fps), (w,h))
    return vw

prev = None
writer = None

if src.is_file():
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3: fps = 25.0
    ok, fr0 = cap.read()
    if not ok or fr0 is None:
        raise RuntimeError("Cannot read first frame.")
    H,W = fr0.shape[:2]
    writer = ensure_writer(save, W, H, fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def iterator():
        while True:
            ok, fr = cap.read()
            if not ok: break
            yield fr
        cap.release()
else:
    frames = sorted(src.glob("*.png"))
    if not frames:
        raise RuntimeError(f"No png frames under: {src}")
    fr0 = cv2.imread(str(frames[0]))
    if fr0 is None:
        raise RuntimeError(f"Cannot read: {frames[0]}")
    H,W = fr0.shape[:2]
    fps = float(fps_arg) if fps_arg else 25.0
    writer = ensure_writer(save, W, H, fps)

    def iterator():
        for p in frames:
            fr = cv2.imread(str(p))
            if fr is None: 
                continue
            yield fr

with torch.no_grad():
    for frame_bgr in iterator():
        H,W = frame_bgr.shape[:2]
        inp = cv2.resize(frame_bgr, (size,size), interpolation=cv2.INTER_LINEAR)
        inp_rgb = inp[:,:,::-1].copy()
        x = torch.from_numpy(inp_rgb).permute(2,0,1).unsqueeze(0).float()/255.0
        x = x.to(device)
        if half: x = x.half()
        with torch.autocast(device_type="cuda", enabled=half):
            logits = net(x)
            prob = torch.sigmoid(logits)[0,0].float().cpu().numpy()

        if prev is None: prob_s = prob
        else: prob_s = smooth*prev + (1.0-smooth)*prob
        prev = prob_s

        prob_up = cv2.resize(prob_s, (W,H), interpolation=cv2.INTER_LINEAR)
        mask = (prob_up >= th).astype(np.uint8)

        overlay = frame_bgr.copy()
        overlay[mask>0] = (0,0,255)
        out = cv2.addWeighted(frame_bgr, 0.65, overlay, 0.35, 0)
        writer.write(out)

writer.release()
print("[OK] Saved video ->", save, f"(th={th}, smooth={smooth})")
PY

info "DONE."
info "1) merged train set: $out_multi/$SPLIT/$CLIP"
info "2) best ckpt: outputs/runs/unet_r34/best.ckpt"
info "3) eval dir: outputs/runs/unet_r34/eval"
info "4) overlay video: $OUT_VIDEO"
