from pathlib import Path
from torch.utils.data import Dataset
import cv2
import numpy as np
from .transforms import train_tf, val_tf

class FrameMaskDataset(Dataset):
    def __init__(self, root, transform='train'):
        self.root = Path(root)
        self.transform = train_tf if transform == 'train' else val_tf
        self.items = []
        for clip in sorted(self.root.iterdir()):
            if not clip.is_dir():
                continue
            fdir, mdir = clip/'frames', clip/'masks'
            if not fdir.exists() or not mdir.exists():
                continue
            for imgp in sorted(fdir.glob('*.png')):
                mp = mdir / imgp.name
                if mp.exists():
                    self.items.append((imgp, mp))
        if not self.items:
            raise RuntimeError(f'No (frame, mask) pairs under {root}')

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, i):
        imgp, mp = self.items[i]
        img = cv2.imread(str(imgp))
        if img is None:
            raise RuntimeError(f"Failed to read image: {imgp}")
        img = img[:, :, ::-1]  # BGR->RGB
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mp}")
        mask = (mask > 0).astype(np.uint8) * 255  # 只要>0都视为前景（适配你的手标）
        aug = self.transform(image=img, mask=mask)
        x = aug['image'].float() / 255.0
        y = (aug['mask'].float() / 255.0).unsqueeze(0)
        return x, y, str(imgp)
