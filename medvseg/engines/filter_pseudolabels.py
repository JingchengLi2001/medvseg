import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from medvseg.utils.common import ensure_dir
from medvseg.utils.flow import warp_mask, iou_binary

def read_mask(p: Path, resize_to: int | None = None):
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if resize_to is not None:
        m = cv2.resize(m, (resize_to, resize_to), interpolation=cv2.INTER_NEAREST)
    m = (m > 127).astype(np.float32)
    return torch.from_numpy(m)[None, None]

def main(pred_root: str, flow_root: str, output_root: str,
         iou_th: float = 0.7, resize: int | None = None):
    pred_root = Path(pred_root)
    flow_root = Path(flow_root)
    output_root = Path(output_root)

    for split in sorted(pred_root.iterdir()):
        if not split.is_dir():
            continue
        for clip in sorted(split.iterdir()):
            fdir = clip / 'frames'
            mdir = clip / 'masks'
            if not fdir.exists() or not mdir.exists():
                continue

            out = output_root / split.name / clip.name
            (out / 'frames').mkdir(parents=True, exist_ok=True)
            (out / 'masks').mkdir(parents=True, exist_ok=True)

            frames = sorted(fdir.glob('*.png'))
            flows_dir = flow_root / split.name / clip.name

            # 复制帧到输出
            for f in frames:
                img = cv2.imread(str(f))
                cv2.imwrite(str(out / 'frames' / f.name), img)

            # 对相邻帧做一致性
            for i in range(len(frames) - 1):
                t = frames[i]
                tp1 = frames[i + 1]
                Mt = read_mask(mdir / t.name, resize)
                Mtp1 = read_mask(mdir / tp1.name, resize)
                if Mt is None or Mtp1 is None:
                    continue
                flow_fwd = np.load(str(flows_dir / f"{t.stem}_to_{tp1.stem}.npy"))
                # 将 M_{t+1} 对齐到 t：需要 backward(t+1->t)，用 -flow_fwd 近似
                Mtp1_warp_to_t = warp_mask(Mtp1, -flow_fwd)
                iou = iou_binary((Mt > 0.5).float(), (Mtp1_warp_to_t > 0.5).float())
                if iou >= iou_th:
                    cv2.imwrite(str(out / 'masks' / t.name),
                                (Mt[0, 0].numpy() * 255).astype('uint8'))
            print('Filtered ->', out)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-root', required=True)
    ap.add_argument('--flow-root', required=True)
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--iou-th', type=float, default=0.7)
    ap.add_argument('--resize', type=int, default=None)
    args = ap.parse_args()
    main(**vars(args))
