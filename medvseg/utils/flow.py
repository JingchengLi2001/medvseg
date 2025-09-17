from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np

def warp_mask(mask_src: torch.Tensor, flow_tgt_to_src: np.ndarray) -> torch.Tensor:
    """
    Backward warping: 将 source 时刻的 mask 按“目标->源”的 backward flow 对齐到 target。
    参数:
      mask_src: (1,1,H,W) torch float32 in [0,1]，表示“源帧”的掩码
      flow_tgt_to_src: (H,W,2) numpy，表示 target 像素在 source 中的采样坐标偏移 (dx, dy)
                       *注意*：这是 backward flow（target -> source）。
                       若你只有 forward flow (source->target)，可用近似：flow_tgt_to_src ≈ -flow_src_to_tgt
    返回:
      对齐到 target 的掩码 (1,1,H,W)
    """
    b, _, h, w = mask_src.shape
    flow = torch.from_numpy(flow_tgt_to_src).to(mask_src.device).float()  # H,W,2

    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=mask_src.device),
                                    torch.arange(w, device=mask_src.device),
                                    indexing='ij')
    tgt = torch.stack((grid_x, grid_y), dim=2).float() + flow  # H,W,2 (x,y)

    tgt_x = 2.0 * (tgt[..., 0] / max(w - 1, 1)) - 1.0
    tgt_y = 2.0 * (tgt[..., 1] / max(h - 1, 1)) - 1.0
    norm_grid = torch.stack((tgt_x, tgt_y), dim=-1).unsqueeze(0)  # 1,H,W,2

    warped = F.grid_sample(mask_src, norm_grid, mode='bilinear',
                           padding_mode='zeros', align_corners=True)
    return warped

def iou_binary(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
    inter = (a * b).sum().item()
    union = (a + b - a * b).sum().item() + eps
    return inter / union
