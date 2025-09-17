import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt

def dice_loss(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3)) + eps
    den = (probs**2 + targets**2).sum(dim=(2, 3)) + eps
    return (1 - num / den).mean()

class BoundaryLoss(nn.Module):
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        b, _, h, w = targets.shape
        dts = []
        for i in range(b):
            t = (targets[i, 0].detach().cpu().numpy() > 0.5)
            dist = distance_transform_edt(t == 0) + distance_transform_edt(t == 1)
            dts.append(torch.from_numpy(dist))
        dt = torch.stack(dts, dim=0).unsqueeze(1).to(logits.device).float()
        return (probs * dt).mean()

class ComboLoss(nn.Module):
    def __init__(self, w_dice=1.0, w_bce=1.0, w_boundary=1.0):
        super().__init__()
        self.wd, self.wb, self.wbd = w_dice, w_bce, w_boundary
        self.bce = nn.BCEWithLogitsLoss()
        self.bd = BoundaryLoss()
    def forward(self, logits, targets):
        loss = 0.0
        if self.wd:  loss += self.wd  * dice_loss(logits, targets)
        if self.wb:  loss += self.wb  * self.bce(logits, targets)
        if self.wbd: loss += self.wbd * self.bd(logits, targets)
        return loss
