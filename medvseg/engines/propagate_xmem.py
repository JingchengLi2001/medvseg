import argparse
from pathlib import Path
import subprocess

from medvseg.utils.common import ensure_dir

def run_xmem_clip(xmem_root: Path, frames_dir: Path, masks_dir: Path, out_dir: Path):
    ensure_dir(out_dir)
    seeds = sorted(masks_dir.glob('*.png'))
    if not seeds:
        print('No seed mask in', masks_dir)
        return
    seed = seeds[0]
    cmd = [
        'python', str(xmem_root/'eval.py'),
        '--model', str(xmem_root/'saves'/'xmem.pth'),
        '--images', str(frames_dir),
        '--mask', str(seed),
        '--output', str(out_dir),
    ]
    print('RUN:', ' '.join(cmd))
    subprocess.run(cmd, check=True)

def main(images_root: str, output_root: str, xmem_root: str):
    images_root = Path(images_root)
    output_root = Path(output_root)
    xmem_root = Path(xmem_root)
    for split in sorted(images_root.iterdir()):
        if not split.is_dir(): continue
        for clip in sorted(split.iterdir()):
            fdir, mdir = clip/'frames', clip/'masks'
            if not fdir.exists() or not mdir.exists(): continue
            out = output_root/split.name/clip.name
            run_xmem_clip(xmem_root, fdir, mdir, out)

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--images-root', required=True)
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--xmem-root', required=True)
    args = ap.parse_args()
    main(**vars(args))
