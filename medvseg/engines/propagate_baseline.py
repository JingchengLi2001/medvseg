import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from medvseg.utils.common import ensure_dir
from medvseg.utils.flow import warp_mask


def farneback_flow(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> np.ndarray:
    """forward flow: t -> t+1 (dx,dy)"""
    g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 25, 5, 7, 1.5, 0)
    return flow.astype(np.float32)  # H,W,2


def _read_bgr(p: Path, resize: int | None):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if resize:
        img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_LINEAR)
    return img


def _read_mask_binary(p: Path, resize: int | None):
    """
    读取单通道 mask，输出 torch float32 (1,1,H,W) in {0,1}
    注意：这里用 >0 判前景，兼容你之前的红色灰度(76) / 255 / 1 等情况
    """
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if resize:
        m = cv2.resize(m, (resize, resize), interpolation=cv2.INTER_NEAREST)
    m = (m > 0).astype(np.float32)
    return torch.from_numpy(m)[None, None]


@torch.no_grad()
def propagate_clip(frames_dir: Path,
                   masks_dir: Path,
                   out_frames_dir: Path,
                   out_masks_dir: Path,
                   out_flow_dir: Path,
                   resize: int = 512,
                   flows_only: bool = False,
                   seed_name: str | None = None):
    ensure_dir(out_frames_dir)
    ensure_dir(out_masks_dir)
    ensure_dir(out_flow_dir)

    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        return

    # 1) 读入并写出 resize 后的 frames（保证后续 frame/mask 尺寸一致）
    imgs = []
    for fp in frames:
        img = _read_bgr(fp, resize)
        if img is None:
            raise RuntimeError(f"Failed to read frame: {fp}")
        imgs.append(img)
        cv2.imwrite(str(out_frames_dir / fp.name), img)

    # 2) 计算并保存所有 forward flows（不依赖种子）
    flows = []
    for i in range(len(imgs) - 1):
        flow_fwd = farneback_flow(imgs[i], imgs[i + 1])  # i -> i+1
        flows.append(flow_fwd)
        a, b = frames[i], frames[i + 1]
        np.save(str(out_flow_dir / f"{a.stem}_to_{b.stem}.npy"), flow_fwd)

    if flows_only:
        return

    # 3) 找到 seed（允许任意帧）
    seed_path = None
    if seed_name is not None:
        cand = masks_dir / seed_name
        if cand.exists():
            seed_path = cand
        else:
            raise RuntimeError(f"seed_name={seed_name} but file not found: {cand}")
    else:
        seeds = sorted(masks_dir.glob("*.png"))
        if seeds:
            seed_path = seeds[0]

    if seed_path is None:
        print("No seed mask found in", masks_dir)
        return

    seed_frame_path = frames_dir / seed_path.name
    if not seed_frame_path.exists():
        raise RuntimeError(
            f"Seed mask name must match a frame name.\n"
            f"mask={seed_path.name} but frame not found under frames/: {seed_frame_path}"
        )

    # 找 seed index
    seed_idx = None
    for i, fp in enumerate(frames):
        if fp.name == seed_path.name:
            seed_idx = i
            break
    if seed_idx is None:
        raise RuntimeError(f"Seed frame {seed_path.name} not found in sorted frames list")

    # 4) 读取 seed mask（binary 0/1）
    m_seed = _read_mask_binary(seed_path, resize)
    if m_seed is None:
        raise RuntimeError(f"Failed to read seed mask: {seed_path}")
    if float(m_seed.sum().item()) <= 0:
        raise RuntimeError(f"Seed mask is empty (all zeros): {seed_path}")

    def save_mask(name: str, m: torch.Tensor):
        outp = out_masks_dir / name
        cv2.imwrite(str(outp), (m[0, 0].cpu().numpy() * 255).astype("uint8"))

    # 5) 写入 seed 本身
    save_mask(frames[seed_idx].name, m_seed)

    # 6) 向后传播：i -> i+1 用 -flow(i) 近似 backward
    m = m_seed.clone()
    for i in range(seed_idx, len(frames) - 1):
        flow_fwd = flows[i]           # i -> i+1
        m = warp_mask(m, -flow_fwd)   # 对齐到 i+1
        save_mask(frames[i + 1].name, m)

    # 7) 向前传播：要得到 i 的 mask（target=i），source=i+1
    #    warp_mask 需要 flow(target->source)=flow(i->i+1)，刚好就是 flows[i]
    m = m_seed.clone()
    for i in range(seed_idx - 1, -1, -1):
        flow_fwd = flows[i]          # i -> i+1
        m = warp_mask(m, flow_fwd)   # 输出对齐到 i
        save_mask(frames[i].name, m)


def main(images_root: str, output_root: str, resize: int = 512, flows_only: bool = False, seed_name: str | None = None):
    images_root = Path(images_root)
    output_root = Path(output_root)

    for split in sorted(images_root.iterdir()):
        if not split.is_dir():
            continue
        for clip in sorted(split.iterdir()):
            fdir, mdir = clip / "frames", clip / "masks"
            if not fdir.exists():
                continue
            if not mdir.exists():
                mdir = clip / "masks"  # 保持原结构

            out_frames = output_root / split.name / clip.name / "frames"
            out_masks  = output_root / split.name / clip.name / "masks"
            out_flows  = output_root / "flows" / split.name / clip.name

            propagate_clip(
                frames_dir=fdir,
                masks_dir=mdir,
                out_frames_dir=out_frames,
                out_masks_dir=out_masks,
                out_flow_dir=out_flows,
                resize=resize,
                flows_only=flows_only,
                seed_name=seed_name
            )
            print("Done ->", split.name, clip.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--flows-only", type=int, default=0)
    ap.add_argument("--seed-name", type=str, default=None, help="optional, e.g. 0280.png")
    args = ap.parse_args()
    main(args.images_root, args.output_root, args.resize, bool(args.flows_only), args.seed_name)
