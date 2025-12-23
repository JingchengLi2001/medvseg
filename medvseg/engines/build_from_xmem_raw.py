import argparse
from pathlib import Path
import shutil

def has_any_pair(root: Path) -> bool:
    if not root.exists():
        return False
    for clip_frames in root.rglob("frames"):
        mdir = clip_frames.parent / "masks"
        if not mdir.exists():
            continue
        for f in clip_frames.glob("*.png"):
            if (mdir / f.name).exists():
                return True
    return False

def build(xmem_root: Path, out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    built = 0

    for split in sorted([d for d in xmem_root.iterdir() if d.is_dir() and d.name != "flows"]):
        for clip in sorted([d for d in split.iterdir() if d.is_dir()]):
            fdir = clip / "frames"
            mdir = clip / "masks"
            if not fdir.exists() or not mdir.exists():
                continue

            frames = {p.name for p in fdir.glob("*.png")}
            masks = {p.name for p in mdir.glob("*.png")}
            inter = sorted(frames & masks)
            if not inter:
                continue

            out = out_root / split.name / clip.name
            (out / "frames").mkdir(parents=True, exist_ok=True)
            (out / "masks").mkdir(parents=True, exist_ok=True)

            for name in inter:
                shutil.copy2(str(fdir / name), str(out / "frames" / name))
                shutil.copy2(str(mdir / name), str(out / "masks" / name))

            built += len(inter)

    return built

def main(xmem_root: str, out_root: str):
    xmem_root = Path(xmem_root)
    out_root = Path(out_root)

    # 如果已有有效 pairs，就不覆盖
    if has_any_pair(out_root):
        print("[INFO] pseudolabels_clean already has valid pairs. skip fallback.")
        return

    print("[WARN] No valid pairs after filtering. Fallback: build dataset from xmem_raw (frames ∩ masks).")
    built = build(xmem_root, out_root)
    print("[OK] Built pairs =", built)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmem-root", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()
    main(args.xmem_root, args.out_root)
