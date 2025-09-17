import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

def load_raft(raft_root: Path, device: str):
    import sys
    sys.path.append(str(raft_root))
    from core.raft import RAFT
    class Args:
        small=False; mixed_precision=False; alternate_corr=False
    model = RAFT(Args())
    weights = raft_root/'models'/'raft-things.pth'
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    from core.utils.utils import InputPadder
    return model, InputPadder

@torch.no_grad()
def infer_pair(model, Padder, img1_rgb, img2_rgb, device):
    I1 = torch.from_numpy(img1_rgb).permute(2,0,1)[None].float()/255.0
    I2 = torch.from_numpy(img2_rgb).permute(2,0,1)[None].float()/255.0
    I1, I2 = I1.to(device), I2.to(device)
    padder = Padder(I1.shape)
    I1, I2 = padder.pad(I1, I2)
    _, flow_up = model(I1, I2, iters=20, test_mode=True)
    return flow_up[0].permute(1,2,0).detach().cpu().numpy()

def main(images_root: str, output_root: str, raft_root: str, device: str='cuda'):
    device = device if (device=='cuda' and torch.cuda.is_available()) else 'cpu'
    model, Padder = load_raft(Path(raft_root), device)

    images_root = Path(images_root)
    for split in sorted(images_root.iterdir()):
        if not split.is_dir(): continue
        for clip in sorted(split.iterdir()):
            fdir = clip/'frames'
            if not fdir.exists(): continue
            out = Path(output_root)/'flows'/split.name/clip.name
            out.mkdir(parents=True, exist_ok=True)
            frames = sorted(fdir.glob('*.png'))
            for i in range(len(frames)-1):
                a,b = frames[i], frames[i+1]
                im1 = cv2.cvtColor(cv2.imread(str(a)), cv2.COLOR_BGR2RGB)
                im2 = cv2.cvtColor(cv2.imread(str(b)), cv2.COLOR_BGR2RGB)
                flow = infer_pair(model, Padder, im1, im2, device)
                np.save(out/f"{a.stem}_to_{b.stem}.npy", flow)
            print('RAFT Flow ->', out)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--images-root', required=True)
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--raft-root', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    main(**vars(args))
