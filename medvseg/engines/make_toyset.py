import argparse
from pathlib import Path
import cv2
import numpy as np

def make(root='data/TOY'):
    out_root = Path(root)/'test_easy_seen'/'clip_0001'
    (out_root/'frames').mkdir(parents=True, exist_ok=True)
    (out_root/'masks').mkdir(parents=True, exist_ok=True)
    H,W = 384, 512
    for t in range(1, 21):
        img = np.full((H,W,3), (30,30,30), np.uint8)
        cx = 120 + t*8; cy = 180 + int(10*np.sin(t/2)); r=35+(t%3)
        cv2.circle(img,(cx,cy), r+8, (70,70,70), -1)
        cv2.circle(img,(cx,cy), r,   (200,200,200), -1)
        cv2.imwrite(str(out_root/'frames'/f"{t:04d}.png"), img)
    mask = np.zeros((H,W), np.uint8)
    cv2.circle(mask,(120+8,180+int(10*np.sin(0.5))), 36, 255, -1)
    cv2.imwrite(str(out_root/'masks'/'0001.png'), mask)
    print('Toyset ->', out_root)

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/TOY')
    args = ap.parse_args()
    make(args.root)
