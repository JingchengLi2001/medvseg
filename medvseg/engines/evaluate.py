import argparse
from pathlib import Path
import torch
import cv2
from torch.utils.data import DataLoader

from medvseg.data.datasets import FrameMaskDataset
from medvseg.models.student_unet import StudentUNet
from medvseg.utils.metrics import dice_score, iou_score
from medvseg.utils.common import ensure_dir

def main(model: str, images_root: str, save_dir: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ensure_dir(save_dir)

    ds = FrameMaskDataset(images_root, transform='val')
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    net = StudentUNet('resnet34', 3, 1)
    state = torch.load(model, map_location=device)
    net.load_state_dict(state['model'])
    net.to(device).eval()

    dices, ious = [], []
    with torch.no_grad():
        for x, y, p in dl:
            x, y = x.to(device), y.to(device)
            pred = net(x)
            dices.append(dice_score(pred, y))
            ious.append(iou_score(pred, y))
            vis = (torch.sigmoid(pred)[0, 0].cpu().numpy() * 255).astype('uint8')
            outp = Path(save_dir) / (Path(p[0]).stem + '_pred.png')
            cv2.imwrite(str(outp), vis)

    print(f"Mean Dice={sum(dices)/len(dices):.4f}, Mean IoU={sum(ious)/len(ious):.4f}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--images-root', required=True)
    ap.add_argument('--save-dir', required=True)
    args = ap.parse_args()
    main(**vars(args))
