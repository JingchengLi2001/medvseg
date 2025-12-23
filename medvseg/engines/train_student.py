import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from medvseg.data.datasets import FrameMaskDataset
from medvseg.models.student_unet import StudentUNet
from medvseg.utils.losses import ComboLoss
from medvseg.utils.metrics import dice_score, iou_score
from medvseg.utils.common import ensure_dir

def train(data_root: str, val_root: str, save_dir: str,
          epochs: int = 5, batch_size: int = 4, lr: float = 1e-3,
          weight_decay: float = 1e-4, num_workers: int = 0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ensure_dir(save_dir)

    train_ds = FrameMaskDataset(data_root, transform='train')
    val_ds = FrameMaskDataset(val_root, transform='val')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = StudentUNet('resnet34', 3, 1).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    loss_fn = ComboLoss(1.0, 1.0, 1.0)

    best = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, ncols=100, desc=f'Epoch {ep}/{epochs}')
        for x, y, _ in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = model(x)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        model.eval()
        dices, ious = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                dices.append(dice_score(pred, y))
                ious.append(iou_score(pred, y))
        md, mi = sum(dices) / len(dices), sum(ious) / len(ious)
        print(f'Val Dice={md:.4f} IoU={mi:.4f}')
        if md > best:
            best = md
            torch.save({'model': model.state_dict()}, Path(save_dir) / 'best.ckpt')
            print('** Saved best')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--val-root', required=True)
    ap.add_argument('--save-dir', required=True)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--num-workers', type=int, default=0)
    args = ap.parse_args()
    train(**vars(args))
