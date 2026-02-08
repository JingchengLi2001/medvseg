import os
import albumentations as A
from albumentations.pytorch import ToTensorV2


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


# Keep backward-compatible defaults:
# - RESIZE defaults to 512 (old behavior).
# - Augmentations default to the previous strong set.
SIZE = int(os.environ.get("RESIZE", "512"))
WEAK_AUG = _truthy(os.environ.get("AUG_WEAK", "0"))
USE_MOTIONBLUR = _truthy(os.environ.get("AUG_MOTIONBLUR", "1"))

_base = [A.Resize(SIZE, SIZE)]

if WEAK_AUG:
    aug = _base + [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.4),
        ToTensorV2(),
    ]
else:
    aug = _base + [
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), rotate=(-10, 10), shear=(-5, 5), p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.HueSaturationValue(10, 10, 10, p=0.3),
        *( [A.MotionBlur(3, p=0.2)] if USE_MOTIONBLUR else [] ),
        ToTensorV2(),
    ]

train_tf = A.Compose(aug, is_check_shapes=False)

val_tf = A.Compose([
    A.Resize(SIZE, SIZE),
    ToTensorV2(),
], is_check_shapes=False)
