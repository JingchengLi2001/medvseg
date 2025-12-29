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
                       注意：这是 backward flow（target -> source）。
    返回:
      对齐到 target 的掩码 (1,1,H,W)
    """
    b, _, h, w = mask_src.shape
    flow = torch.from_numpy(flow_tgt_to_src).to(mask_src.device).float()  # H,W,2

    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=mask_src.device),
        torch.arange(w, device=mask_src.device),
        indexing='ij'
    )
    tgt = torch.stack((grid_x, grid_y), dim=2).float() + flow  # H,W,2 (x,y)

    tgt_x = 2.0 * (tgt[..., 0] / max(w - 1, 1)) - 1.0
    tgt_y = 2.0 * (tgt[..., 1] / max(h - 1, 1)) - 1.0
    norm_grid = torch.stack((tgt_x, tgt_y), dim=-1).unsqueeze(0)  # 1,H,W,2

    warped = F.grid_sample(mask_src, norm_grid, mode='nearest',
                           padding_mode='zeros', align_corners=True)
    return warped

def iou_binary(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
    inter = (a * b).sum().item()
    union = (a + b - a * b).sum().item() + eps
    return inter / union

def _grid_sample_2ch(field_1x2xHxW: torch.Tensor, coords_1xHxWx2: torch.Tensor) -> torch.Tensor:
    """
    field: (1,2,H,W)
    coords: (1,H,W,2) in pixel coords (x,y)
    return: (1,2,H,W) sampled
    """
    _, _, h, w = field_1x2xHxW.shape
    x = coords_1xHxWx2[..., 0]
    y = coords_1xHxWx2[..., 1]
    x_norm = 2.0 * (x / max(w - 1, 1)) - 1.0
    y_norm = 2.0 * (y / max(h - 1, 1)) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)
    return F.grid_sample(field_1x2xHxW, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)

def fb_consistency_conf(flow_fwd: np.ndarray, flow_bwd: np.ndarray,
                        sigma: float = 2.0, device: str | None = None) -> np.ndarray:
    """
    Forward-backward consistency confidence.
    err(p) = || fwd(p) + bwd(p + fwd(p)) ||
    conf = exp(-err / sigma)
    return conf: (H,W) float32 in [0,1]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    fwd = torch.from_numpy(flow_fwd).permute(2, 0, 1).unsqueeze(0).float().to(device)
    bwd = torch.from_numpy(flow_bwd).permute(2, 0, 1).unsqueeze(0).float().to(device)

    _, _, h, w = fwd.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=device),
                            torch.arange(w, device=device), indexing="ij")
    base = torch.stack([xx, yy], dim=0).float()

    coords = (base + fwd[0]).permute(1, 2, 0).unsqueeze(0)
    bwd_at = _grid_sample_2ch(bwd, coords)

    fb = fwd + bwd_at
    err = torch.norm(fb, dim=1)
    conf = torch.exp(-err / float(sigma))
    return conf.squeeze(0).detach().cpu().numpy().astype(np.float32)

def iou_weighted(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> float:
    """
    a,b,w: (1,1,H,W) float tensors
    """
    inter = (a * b * w).sum().item()
    union = ((a + b - a * b) * w).sum().item() + eps
    return inter / union
