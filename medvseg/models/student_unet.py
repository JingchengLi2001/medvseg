from __future__ import annotations

import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class StudentUNet(nn.Module):
    """Lightweight UNet student.

    Environment knobs (all optional, default keeps old behavior):
      - USE_IMAGENET=0/1          : use ImageNet pretrained encoder weights
      - DECODER_ATTENTION=0/1     : enable decoder attention (scSE)
      - IMAGENET_NORM=0/1         : apply ImageNet mean/std normalization in forward
                                  (defaults to 1 when USE_IMAGENET=1 and in_channels==3)
    """

    def __init__(self, backbone: str = "resnet34", in_channels: int = 3, classes: int = 1):
        super().__init__()

        def _truthy(v: str) -> bool:
            return v.strip().lower() in ("1", "true", "yes", "y", "on")

        # Backward/compat env names:
        #   USE_IMAGENET=1 OR ENCODER_WEIGHTS=imagenet
        #   DECODER_ATTENTION=1 OR DECODER_ATTENTION=scse
        use_imagenet = _truthy(os.environ.get("USE_IMAGENET", "0"))
        enc_w = os.environ.get("ENCODER_WEIGHTS", "").strip().lower()
        if enc_w == "imagenet":
            use_imagenet = True

        dec_raw = os.environ.get("DECODER_ATTENTION", "0").strip().lower()
        dec_attn = _truthy(dec_raw) or (dec_raw in ("scse", "se", "squeeze", "squeezeexcite"))

        # Pretrained weights only make sense for RGB input.
        if in_channels != 3:
            use_imagenet = False

        encoder_weights = "imagenet" if use_imagenet else None
        decoder_attention_type = "scse" if dec_attn else None

        self.net = smp.Unet(
            encoder_name=backbone,
            encoder_weights=encoder_weights,   # offline-safe default: None
            in_channels=in_channels,
            classes=classes,
            decoder_attention_type=decoder_attention_type,
        )

        # Optional normalization (recommended when using ImageNet weights)
        if use_imagenet and in_channels == 3:
            do_norm = os.environ.get("IMAGENET_NORM", "1").strip() in ("1", "true", "True")
        else:
            do_norm = os.environ.get("IMAGENET_NORM", "0").strip() in ("1", "true", "True")
        self.do_norm = bool(do_norm)

        if in_channels == 3:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("_mean", mean)
            self.register_buffer("_std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.do_norm and hasattr(self, "_mean") and hasattr(self, "_std"):
            x = (x - self._mean) / self._std
        return self.net(x)
