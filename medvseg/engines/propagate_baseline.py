import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from medvseg.utils.common import ensure_dir
from medvseg.utils.flow import warp_mask

def farneback_flow(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> np.ndarray:
    """返回 forward flow: t -> t+1 (dx,dy)"""
    g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 25, 5, 7, 1.5, 0)
    return flow.astype(np.float32)  # H,W,2

@torch.no_grad()
def propagate_clip(frames_dir: Path,
                   masks_dir: Path,
                   out_mask_dir: Path,
                   out_flow_dir: Path,
                   resize: int = 512,
                   flows_only: bool = False):
    ensure_dir(out_mask_dir)
    ensure_dir(out_flow_dir)
    frames = sorted(frames_dir.glob('*.png'))
    if not frames:
        return

    seed_path = masks_dir / frames[0].name
    if not seed_path.exists():
        print('No seed mask for', frames_dir)
        return

    def read_rgb(p: Path):
        img = cv2.imread(str(p))
        if resize:
            img = cv2.resize(img, (resize, resize))
        return img

    def read_mask(p: Path):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        if resize:
            m = cv2.resize(m, (resize, resize), interpolation=cv2.INTER_NEAREST)
        m = (m > 127).astype(np.float32)
        return torch.from_numpy(m)[None, None]

    prev_img = read_rgb(frames[0])
    m_prev = read_mask(seed_path)
    if (not flows_only) and (m_prev is not None):
        cv2.imwrite(str(out_mask_dir / frames[0].name),
                    (m_prev[0, 0].numpy() * 255).astype('uint8'))

    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        img2 = read_rgb(b)
        flow_fwd = farneback_flow(prev_img, img2)  # t -> t+1
        np.save(str(out_flow_dir / f"{a.stem}_to_{b.stem}.npy"), flow_fwd)

        if not flows_only and m_prev is not None:
            # 需要 backward flow (t+1->t) 来对齐；用 -forward 近似
            m_next = warp_mask(m_prev, -flow_fwd)  # 对齐到 t+1
            cv2.imwrite(str(out_mask_dir / b.name),
                        (m_next[0, 0].numpy() * 255).astype('uint8'))
            m_prev = m_next.clone()

        prev_img = img2

def main(images_root: str, output_root: str, resize: int = 512, flows_only: bool = False):
    images_root = Path(images_root)
    output_root = Path(output_root)
    for split in sorted(images_root.iterdir()):
        if not split.is_dir():
            continue
        for clip in sorted(split.iterdir()):
            fdir, mdir = clip / 'frames', clip / 'masks'
            if not fdir.exists() or not mdir.exists():
                continue
            out_mask = output_root / split.name / clip.name / 'masks'
            out_flow = output_root / 'flows' / split.name / clip.name
            propagate_clip(fdir, mdir, out_mask, out_flow, resize, flows_only)
            print('Done ->', split.name, clip.name)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--images-root', required=True)
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--resize', type=int, default=512)
    ap.add_argument('--flows-only', type=int, default=0)
    args = ap.parse_args()
    main(args.images_root, args.output_root, args.resize, bool(args.flows_only))
