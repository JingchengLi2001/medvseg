
from __future__ import annotations
import math
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np

def _mad(x: np.ndarray) -> float:
    """Median absolute deviation (MAD)."""
    x = np.asarray(x, dtype=np.float32)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return mad

def robust_threshold_high(x: np.ndarray, k: float = 4.0) -> float:
    """Threshold for 'too high is bad': med + k * 1.4826 * MAD."""
    x = np.asarray(x, dtype=np.float32)
    med = float(np.median(x))
    mad = _mad(x)
    return med + k * 1.4826 * mad

def robust_threshold_low(x: np.ndarray, k: float = 4.0) -> float:
    """Threshold for 'too low is bad': med - k * 1.4826 * MAD."""
    x = np.asarray(x, dtype=np.float32)
    med = float(np.median(x))
    mad = _mad(x)
    return med - k * 1.4826 * mad

def farneback_flow(gray1: np.ndarray, gray2: np.ndarray) -> np.ndarray:
    """Dense optical flow (Farneback). Returns float32 HxWx2 (dx,dy)."""
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=25, iterations=5,
        poly_n=7, poly_sigma=1.5, flags=0
    )
    return flow.astype(np.float32)

def fb_confidence(flow_fwd: np.ndarray, flow_bwd: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Forward-backward consistency confidence (numpy version).
    err(p) = || fwd(p) + bwd(p + fwd(p)) ||, conf = exp(-err / sigma)
    """
    h, w = flow_fwd.shape[:2]
    # grid of target coords in (x,y)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x2 = xs + flow_fwd[..., 0]
    y2 = ys + flow_fwd[..., 1]

    # sample bwd at (x2,y2)
    bwd_x = cv2.remap(flow_bwd[..., 0], x2, y2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    bwd_y = cv2.remap(flow_bwd[..., 1], x2, y2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    err = np.sqrt((flow_fwd[..., 0] + bwd_x) ** 2 + (flow_fwd[..., 1] + bwd_y) ** 2)
    conf = np.exp(-err / max(sigma, 1e-6)).astype(np.float32)
    return np.clip(conf, 0.0, 1.0)

def blur_var(gray: np.ndarray) -> float:
    """Variance of Laplacian (low -> blurry)."""
    v = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(v)

def motion_metrics(prev_bgr: np.ndarray, curr_bgr: np.ndarray,
                   size: int = 256,
                   conf_sigma: float = 2.0,
                   conf_thr: float = 0.6) -> Dict[str, float]:
    """
    Compute motion/quality metrics on downsampled frames.
    Returns:
      flow_mean, flow_p95, conf_mean, conf_pct, blur_var
    """
    a = cv2.resize(prev_bgr, (size, size), interpolation=cv2.INTER_AREA)
    b = cv2.resize(curr_bgr, (size, size), interpolation=cv2.INTER_AREA)
    g1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    fwd = farneback_flow(g1, g2)
    bwd = farneback_flow(g2, g1)
    mag = np.sqrt(fwd[..., 0] ** 2 + fwd[..., 1] ** 2).astype(np.float32)

    p95 = float(np.percentile(mag, 95))
    mean = float(mag.mean())

    conf = fb_confidence(fwd, bwd, sigma=conf_sigma)
    conf_mean = float(conf.mean())
    conf_pct = float((conf > conf_thr).mean())

    bv = blur_var(g2)
    return {
        "flow_mean": mean,
        "flow_p95": p95,
        "conf_mean": conf_mean,
        "conf_pct": conf_pct,
        "blur_var": bv,
    }

def auto_calibrate(metrics: List[Dict[str, float]],
                   stable_quantile: float = 0.6,
                   k_mad: float = 4.0) -> Dict[str, float]:
    """
    Auto-calibrate per-video thresholds using a stable subset:
    - choose frames with flow_p95 in the lower stable_quantile as "stable"
    - estimate thresholds via median +/- k*MAD on that subset
    """
    if not metrics:
        # safe fallbacks
        return {
            "flow_p95_th": 8.0,
            "conf_mean_th": 0.55,
            "conf_pct_th": 0.15,
            "blur_var_th": 15.0,
        }

    flow_p95 = np.array([m["flow_p95"] for m in metrics], dtype=np.float32)
    q = float(np.quantile(flow_p95, stable_quantile))
    stable_idx = flow_p95 <= q
    if stable_idx.sum() < 10:  # too few, fallback to all
        stable_idx = np.ones_like(flow_p95, dtype=bool)

    def sel(key: str) -> np.ndarray:
        arr = np.array([m[key] for m in metrics], dtype=np.float32)
        return arr[stable_idx]

    flow_p95_s = sel("flow_p95")
    conf_mean_s = sel("conf_mean")
    conf_pct_s = sel("conf_pct")
    blur_s = sel("blur_var")

    flow_p95_th = robust_threshold_high(flow_p95_s, k=k_mad)
    conf_mean_th = robust_threshold_low(conf_mean_s, k=k_mad)
    conf_pct_th = robust_threshold_low(conf_pct_s, k=k_mad)
    blur_var_th = robust_threshold_low(blur_s, k=k_mad)

    # clamp to sane ranges
    conf_mean_th = float(np.clip(conf_mean_th, 0.15, 0.95))
    conf_pct_th = float(np.clip(conf_pct_th, 0.02, 0.95))
    blur_var_th = float(max(3.0, blur_var_th))
    flow_p95_th = float(max(0.5, flow_p95_th))

    return {
        "flow_p95_th": flow_p95_th,
        "conf_mean_th": conf_mean_th,
        "conf_pct_th": conf_pct_th,
        "blur_var_th": blur_var_th,
    }

@dataclass
class GateConfig:
    # downsample size for motion detection
    size: int = 256
    conf_sigma: float = 2.0
    conf_thr: float = 0.6
    # temporal gating
    k_on: int = 3  # require k_on consecutive stable frames to enable segmentation
    # auto calibration
    auto: bool = True
    calib_seconds: float = 3.0
    stable_quantile: float = 0.6
    k_mad: float = 4.0
    # target time-step for fps normalization (seconds)
    dt_time: float = 1.0 / 25.0

class MotionGate:
    """
    State machine:
      - start disabled
      - if stable for k_on consecutive steps -> enable
      - if break -> disable immediately and reset counts
    """
    def __init__(self, cfg: GateConfig):
        self.cfg = cfg
        self.enabled = False
        self.stable_count = 0
        self.th = None  # calibrated thresholds dict

        self._buf: Deque[np.ndarray] = deque()

    def reset(self):
        self.enabled = False
        self.stable_count = 0
        self._buf.clear()

    def set_thresholds(self, th: Dict[str, float]):
        self.th = th

    def dt_frames(self, fps: float) -> int:
        if fps <= 1e-6:
            return 1
        return max(1, int(round(fps * self.cfg.dt_time)))

    def push_frame(self, frame_bgr: np.ndarray):
        """Keep a history buffer for dt_frames pairing."""
        self._buf.append(frame_bgr)
        # cap buffer length
        if len(self._buf) > 20:
            self._buf.popleft()

    def step(self, curr_bgr: np.ndarray, fps: float) -> Tuple[bool, Dict[str, float]]:
        """
        Returns (break, metrics) for current frame.
        Uses frame dt in the past for fps-normalized motion.
        """
        dt = self.dt_frames(fps)
        self.push_frame(curr_bgr)
        if len(self._buf) <= dt:
            # not enough history
            self.stable_count += 1
            if self.stable_count >= self.cfg.k_on:
                self.enabled = True
            return (False, {"warmup": 1.0})

        prev = list(self._buf)[-dt-1]
        met = motion_metrics(prev, curr_bgr, size=self.cfg.size, conf_sigma=self.cfg.conf_sigma, conf_thr=self.cfg.conf_thr)

        if self.th is None:
            # no thresholds -> conservative defaults
            th = {
                "flow_p95_th": 8.0,
                "conf_mean_th": 0.55,
                "conf_pct_th": 0.15,
                "blur_var_th": 15.0,
            }
        else:
            th = self.th

        is_break = (
            (met["flow_p95"] > th["flow_p95_th"]) or
            (met["conf_mean"] < th["conf_mean_th"]) or
            (met["conf_pct"] < th["conf_pct_th"]) or
            (met["blur_var"] < th["blur_var_th"])
        )

        if is_break:
            self.enabled = False
            self.stable_count = 0
        else:
            self.stable_count += 1
            if self.stable_count >= self.cfg.k_on:
                self.enabled = True

        met = {**met, "enabled": float(self.enabled), "stable_count": float(self.stable_count)}
        return is_break, met
