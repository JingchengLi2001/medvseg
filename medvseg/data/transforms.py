import albumentations as A
from albumentations.pytorch import ToTensorV2

train_tf = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.Affine(scale=(0.9,1.1), rotate=(-10,10), shear=(-5,5), p=0.5),
    A.RandomBrightnessContrast(0.2,0.2,p=0.5),
    A.HueSaturationValue(10,10,10,p=0.3),
    A.MotionBlur(3,p=0.2),
    ToTensorV2()
], is_check_shapes=False)

val_tf = A.Compose([
    A.Resize(512, 512),
    ToTensorV2()
], is_check_shapes=False)

