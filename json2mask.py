import os, glob
import numpy as np
import imageio.v2 as iio
from labelme import utils

def json_to_mask(json_path: str, out_path: str):
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_path = os.path.join(os.path.dirname(json_path), data["imagePath"])
    img = iio.imread(img_path)
    h, w = img.shape[:2]

    # 只吃 polygon / rectangle / circle 等能形成区域的 shape
    shapes = data["shapes"]
    lbl, _ = utils.shapes_to_label(img_shape=(h, w, 3), shapes=shapes, label_name_to_value={"_background_": 0})

    binmask = (lbl > 0).astype(np.uint8) * 255
    iio.imwrite(out_path, binmask)

if __name__ == "__main__":
    for jp in glob.glob("*.json"):
        out = os.path.splitext(jp)[0] + "_mask.png"
        json_to_mask(jp, out)
        print("OK:", jp, "->", out)
