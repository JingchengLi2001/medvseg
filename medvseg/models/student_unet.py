import torch.nn as nn
import segmentation_models_pytorch as smp

class StudentUNet(nn.Module):
    def __init__(self, backbone='resnet34', in_channels=3, classes=1):
        super().__init__()
        self.net = smp.Unet(
            encoder_name=backbone,
            encoder_weights=None,   # 离线稳定：不下载 imagenet 权重
            in_channels=in_channels,
            classes=classes
        )

    def forward(self, x):
        return self.net(x)
