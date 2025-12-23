import argparse
from pathlib import Path
import torch, cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from medvseg.models.student_unet import StudentUNet

def main(model: str, frames_dir: str, out_dir: str, resize: int = 512, thr: float = 0.5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = StudentUNet('resnet34', 3, 1)
    state = torch.load(model, map_location=device)
    net.load_state_dict(state['model'])
    net.to(device).eval()

    tf = A.Compose([A.Resize(resize, resize), ToTensorV2()], is_check_shapes=False)

    frames = sorted(Path(frames_dir).glob('*.png'))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for f in frames:
        img_bgr = cv2.imread(str(f))
        h, w = img_bgr.shape[:2]
        img_rgb = img_bgr[:, :, ::-1]
        x = tf(image=img_rgb)['image'].float()/255.0
        with torch.no_grad():
            y = torch.sigmoid(net(x[None].to(device)))[0,0].cpu().numpy()
        y = (y >= thr).astype('uint8')*255
        y = cv2.resize(y, (w, h), interpolation=cv2.INTER_NEAREST)
        # 可视化叠加
        overlay = img_bgr.copy()
        overlay[y>0] = (0,0,255)  # 红色
        vis = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)
        cv2.imwrite(str(Path(out_dir)/f.name), vis)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--frames-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--resize', type=int, default=512)
    ap.add_argument('--thr', type=float, default=0.5)
    args = ap.parse_args()
    main(**vars(args))
