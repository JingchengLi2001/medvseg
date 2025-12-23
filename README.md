
# MedVSeg：低标注成本的内窥镜视频分割工程（Seed 多帧标注 → 传播 → 筛选 → 训练 → 视频输出）

## 0. 摘要

MedVSeg 是一个面向内窥镜医学视频（息肉、炎症、早癌病灶等）的分割工程原型，目标是在**极低标注成本**的前提下完成**整段视频的病灶区域分割**。工程支持：
- **单帧或多帧手工标注**（Seed masks）
- 基于光流/时序一致性的**伪标签传播与筛选**
- 训练轻量学生分割网络（U-Net/ResNet34）
- 输出逐帧预测结果与可视化视频（overlay/realtime）

该仓库更强调“从原始 mp4 到可复现输出视频”的工程闭环，同时保留清晰的模块边界以支持后续替换传播器（XMem/RAFT/SAM2 等）、增强筛选策略与实时化推理优化。

---

## 1. 背景与动机

### 1.1 问题背景
内窥镜视频中病灶分割用于定量评估（面积、边界、形态变化）、术中提示与后处理分析。理想情况下需要对视频逐帧提供像素级标注，但逐帧人工分割代价极高，难以规模化。

### 1.2 核心挑战
- 视频帧数多（10–60 秒视频在 25fps 下可达数百到上千帧）
- 病灶形态变化、遮挡、镜头抖动、光照变化明显
- 单帧标注传播易发生漂移（drift），伪标签噪声影响训练
- 需要兼顾工程可跑通、可复现与后续可优化空间

### 1.3 本工程的策略
以“少量 Seed 标注”为起点，构建伪标签数据集并训练学生网络：
1. 手工标注少量帧（单帧可跑通，多帧更稳）
2. 对相邻帧计算光流并 warp mask，生成全视频伪标签
3. 用时序一致性指标（IoU）筛除不可靠伪标签，得到训练集
4. 训练 U-Net 类学生分割网络以提升整体鲁棒性与推理速度
5. 对原视频逐帧推理并合成分割视频输出

---

## 2. 方法概览（Pipeline）

### 2.1 数据流（从 mp4 到 overlay.mp4）

1. **视频抽帧**：`myvideo.mp4` → `frames/%04d.png`
2. **Seed 标注**：在若干关键帧上人工生成 `masks/<same_name>.png`  
3. **传播（Propagation）**：用光流将 mask 在时间轴上扩散到其它帧  
4. **筛选（Filtering）**：用相邻帧一致性（warp+IoU）保留可靠帧  
5. **训练（Train student）**：用筛选后的 `frames+masks` 训练 U-Net  
6. **推理与合成视频**：输出 per-frame prediction + overlay，合成 mp4

### 2.2 关于 Seed（单帧/多帧）
- Seed 可以是任意帧，不要求第一帧出现病灶
- 支持标注多帧：用于分段传播与降低漂移（后续优化重点）
- mask 的“显示颜色”不重要；训练与传播使用像素值
  - 二分类推荐：mask 为单通道 0/255（背景=0，前景=255）

---

## 3. 目录结构（工程代码 + 数据 + 输出）

```text
medvseg_unix/
├─ run.sh
├─ requirements.txt
├─ data/
│  ├─ TOY/
│  │  └─ test_easy_seen/clip_0001/{frames,masks}/...
│  └─ MED/
│     ├─ raw/
│     │  └─ myvideo.mp4
│     └─ test_easy_seen/
│        └─ clip_0001/
│           ├─ frames/            # 抽帧 PNG 序列（%04d.png）
│           └─ masks/             # 手工标注帧（可 1 张或多张，文件名与 frames 对齐）
├─ medvseg/
│  ├─ __init__.py
│  ├─ utils/
│  │  ├─ common.py
│  │  ├─ flow.py
│  │  ├─ losses.py
│  │  └─ metrics.py
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ transforms.py
│  │  └─ datasets.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  └─ student_unet.py
│  └─ engines/
│     ├─ __init__.py
│     ├─ make_toyset.py
│     ├─ propagate_baseline.py
│     ├─ filter_pseudolabels.py
│     ├─ train_student.py
│     ├─ evaluate.py
│     ├─ propagate_xmem.py              # 可选（需要 third_party/XMem + 权重）
│     └─ run_raft_inproc.py             # 可选（需要 third_party/RAFT + 权重）
└─ outputs/
   ├─ xmem_raw/
   │  ├─ test_easy_seen/clip_0001/
   │  │  ├─ masks/                      # 传播得到的逐帧伪掩码
   │  │  └─ frames/                     # 传播阶段可能同步的 frames（用于筛选）
   │  └─ flows/test_easy_seen/clip_0001/ # 相邻帧光流 npy
   ├─ pseudolabels_clean/
   │  └─ test_easy_seen/clip_0001/{frames,masks}/...
   └─ runs/
      └─ unet_r34/
         ├─ best.ckpt
         └─ eval/
            ├─ 0001_pred.png
            ├─ ...
            └─ 0xxx_pred.png
````

---

## 4. 代码模块说明（Engineering Architecture）

### 4.1 `medvseg/utils`

* `common.py`

  * `read_yaml(path)`: 读取 YAML
  * `ensure_dir(p)`: 确保目录存在
* `flow.py`

  * `warp_mask(mask_src, flow_tgt_to_src)`: 用 backward flow 将源 mask warp 到目标帧
  * `iou_binary(a,b)`: 二值 IoU
* `losses.py`

  * `dice_loss(logits, targets)`
  * `BoundaryLoss`: 基于 distance_transform_edt 的边界相关损失
  * `ComboLoss`: Dice + BCE + Boundary 的加权组合
* `metrics.py`

  * `dice_score(logits, targets, thr)`
  * `iou_score(logits, targets, thr)`

### 4.2 `medvseg/data`

* `transforms.py`

  * Albumentations 数据增强：Resize、Flip、Affine、Brightness/Contrast 等
  * 关键约束：Albumentations 在执行增强前检查 image/mask 形状一致性
* `datasets.py`

  * `FrameMaskDataset(root, transform)`
  * 目录约定：`root/clip_xxxx/{frames,masks}/*.png`
  * 配对逻辑：同名 `frames/<name>.png` 与 `masks/<name>.png` 同时存在才计入样本
  * 输出：

    * image：float32, [3,H,W], 归一化到 [0,1]
    * mask：float32, [1,H,W], 值为 {0,1}

### 4.3 `medvseg/models`

* `student_unet.py`

  * `segmentation_models_pytorch.Unet(encoder=resnet34)`
  * 注意：离线环境建议 `encoder_weights=None`，避免下载 imagenet 权重

### 4.4 `medvseg/engines`

* `make_toyset.py`

  * 生成 toy 数据：20 帧 + 1 张种子 mask
* `propagate_baseline.py`

  * `farneback_flow(img1,img2)`：相邻帧 forward flow
  * `propagate_clip(...)`：用 flow warp mask 序列并保存，同时保存 flow
  * 支持：

    * `--resize`: 将 frames/masks 统一 resize（保证下游一致）
    * `--flows-only`: 只生成 flow，不生成 masks
    * `--seed-name`: 指定种子帧文件名（支持病灶不在第一帧）
* `filter_pseudolabels.py`

  * 读取传播输出的 frames/masks 与 flows
  * 将 `M_{t+1}` warp 回 `t` 与 `M_t` 计算 IoU
  * IoU ≥ 阈值则保留 `t` 的 mask（输出到 `outputs/pseudolabels_clean/...`）
* `train_student.py`

  * 训练 U-Net 学生网络，输出 `best.ckpt`
* `evaluate.py`

  * 在数据集上推理输出 `*_pred.png`（灰度概率图）

> 说明：若需要端到端“视频输出”，通常增加一个推理脚本（如 `stream_realtime.py`），读取视频或帧序列，逐帧预测并写出 overlay 视频。该模块若尚未存在，可作为后续扩展模块加入 `medvseg/engines/`。

---

## 5. 环境与依赖

### 5.1 Python 与 CUDA

* Python 3.10（conda 环境 `myconda`）
* Torch（GPU 环境，示例：cu118）
* OpenCV（示例：4.10.0.84）

### 5.2 必要 Python 包（训练与推理）

* `numpy`, `scipy`, `opencv-python-headless`
* `albumentations`, `albucore`
* `segmentation-models-pytorch`（训练 U-Net 必需）
* `tqdm`
* 其它：`scikit-learn`, `pandas`（如需）

### 5.3 ffmpeg

用于抽帧与合成视频：

* Debian/Ubuntu：`apt-get install -y ffmpeg`
* 验证：`ffmpeg -version`

---

## 6. 数据准备规范（MED 视频数据）

### 6.1 放置原视频

```text
data/MED/raw/myvideo.mp4
```

### 6.2 抽帧（示例 25fps）

```bash
mkdir -p data/MED/test_easy_seen/clip_0001/frames
ffmpeg -hide_banner -y -i data/MED/raw/myvideo.mp4 \
  -vf fps=25 data/MED/test_easy_seen/clip_0001/frames/%04d.png
```

### 6.3 放置手工标注（单帧或多帧）

```text
data/MED/test_easy_seen/clip_0001/masks/0280.png
# 可选增加更多：
data/MED/test_easy_seen/clip_0001/masks/0300.png
data/MED/test_easy_seen/clip_0001/masks/0350.png
...
```

### 6.4 关键约束（避免训练阶段致命错误）

* 同名配对：mask 文件名必须与 frames 中对应帧一致
* 尺寸一致：mask 与 frame 必须同 H×W
* 二值化：mask 为单通道 0/255（背景 0，前景 255）

---

## 7. 运行流程（准确命令）

### 7.1 一次性全流程（传播 → flow → 筛选 → 训练 → 评测）

```bash
# 激活环境
conda activate myconda
cd ~/medvseg_unix

# 清理旧输出（避免混入历史错误结果）
rm -rf outputs/xmem_raw outputs/pseudolabels_clean outputs/runs/unet_r34

# 传播：指定种子帧（支持病灶不在第一帧）
python -m medvseg.engines.propagate_baseline \
  --images-root data/MED \
  --output-root outputs/xmem_raw \
  --resize 512 \
  --seed-name 0280.png

# 光流：只计算 flows
python -m medvseg.engines.propagate_baseline \
  --images-root data/MED \
  --output-root outputs/xmem_raw \
  --resize 512 \
  --flows-only 1

# 筛选：时序一致性筛选伪标签
python -m medvseg.engines.filter_pseudolabels \
  --pred-root outputs/xmem_raw \
  --flow-root outputs/xmem_raw/flows \
  --output-root outputs/pseudolabels_clean \
  --iou-th 0.4

# 训练：FrameMaskDataset 期望 root 下一层是 clip 目录
python -m medvseg.engines.train_student \
  --data-root outputs/pseudolabels_clean/test_easy_seen \
  --val-root  outputs/pseudolabels_clean/test_easy_seen \
  --epochs 5 --batch-size 4 --lr 1e-3 \
  --save-dir outputs/runs/unet_r34

# 评测：输出每帧预测概率图（灰度）
python -m medvseg.engines.evaluate \
  --model outputs/runs/unet_r34/best.ckpt \
  --images-root outputs/pseudolabels_clean/test_easy_seen \
  --save-dir outputs/runs/unet_r34/eval
```

### 7.2 输出位置

* 传播 masks：`outputs/xmem_raw/test_easy_seen/clip_0001/masks/*.png`
* 筛选后训练集：`outputs/pseudolabels_clean/test_easy_seen/clip_0001/{frames,masks}/*.png`
* 最佳权重：`outputs/runs/unet_r34/best.ckpt`
* 推理输出（单帧预测）：`outputs/runs/unet_r34/eval/*_pred.png`

---

## 8. 从“帧预测”到“分割视频输出”

### 8.1 输出内容说明

`evaluate.py` 产生的 `*_pred.png` 为模型的概率输出（0–255 灰度），通常需要阈值化并叠加到原始帧，再用 ffmpeg 合成视频。

### 8.2 标准合成方式（概率图 → overlay PNG → mp4）

假设：

* 原始帧（resize 后）位于：`outputs/pseudolabels_clean/test_easy_seen/clip_0001/frames`
* 预测图位于：`outputs/runs/unet_r34/eval`
  则通常执行：

1. 生成 overlay PNG 序列到 `outputs/videos/overlay_frames/%04d.png`
2. ffmpeg 合成 `outputs/videos/overlay.mp4`

> 工程建议：将 overlay 生成封装为 `medvseg/engines/render_video.py` 或 `stream_realtime.py`，以便 run.sh 一键输出视频（见第 10 节优化方向）。

---

## 9. 训练日志与指标解释

### 9.1 loss

训练阶段输出的 `loss=...` 来自 `ComboLoss`：

* BCEWithLogitsLoss（像素级二分类）
* Dice loss（区域重叠）
* Boundary loss（边界相关，基于距离变换）

loss 越低通常表示与训练标签（伪标签）一致性更强。

### 9.2 Dice 与 IoU

* Dice（F1）：

  * `Dice = 2TP / (2TP + FP + FN)`
* IoU（Jaccard）：

  * `IoU = TP / (TP + FP + FN)`

Dice 通常比 IoU 更“宽容”，IoU 对边界偏移更敏感。

### 9.3 指标的解释边界

当前工程默认使用伪标签训练与验证，因此日志中的 Val Dice/IoU 衡量的是：

* 学生网络对伪标签分布的拟合程度
  而非严格意义上的“真实 GT 精度”。

真实精度评估通常需要：

* 额外人工标注若干帧作为纯测试集（不参与训练与筛选）

---

## 10. 常见故障根因与工程约束（面向复现与稳定性）

### 10.1 image/mask 尺寸不一致导致 Albumentations 报错

Albumentations 在执行增强前会检查 `image.shape == mask.shape`，若不一致直接抛异常。此类问题常见根因：

* 传播阶段使用 `--resize 512` 输出 masks 为 512×512
* frames 仍为原始分辨率（如 1000×1170）
  解决策略：
* 传播与筛选阶段统一 frames/masks resize（保证进入 Dataset 前一致）

### 10.2 mask 全黑导致训练输出全黑

若 `outputs/xmem_raw` 或 `outputs/pseudolabels_clean` 中 masks 全部为空（nonzero=0），则：

* 训练会退化为“全背景预测”
* 输出 pred 全黑
  根因可能包括：
* seed mask 实际未二值化或阈值后全 0
* seed 与 frame 文件名不匹配导致未读取到有效 seed
* warp/resize 过程中前景丢失
  工程策略：
* 在传播前执行 seed overlay 检查（可写入 `outputs/seed_check/`）

### 10.3 Dataset root 指定错误导致 No pairs

`FrameMaskDataset` 的 root 要求：

* root 下一级为 clip 目录（例如 `clip_0001`）
  因此训练参数建议使用：
* `outputs/pseudolabels_clean/test_easy_seen`
  而非：
* `outputs/pseudolabels_clean`

---

## 11. 多帧标注（重要扩展点）

本工程并不限制只标注单帧。支持与建议：

* 标注多帧（例如每隔 1–2 秒一帧或挑关键视角变化点）
* 传播策略可以扩展为：

  * 分段传播：每段从最近 seed 向两侧传播
  * 多 seed 融合：同一帧来自多个 seed 的预测做融合与一致性筛选
* 预期收益：

  * 明显降低漂移
  * 提升复杂运动/遮挡情况下的稳定性
  * 改善最终训练集质量与学生网络泛化

---

## 12. 未来优化方向（毕业设计可写入“改进点/创新点”）

### 12.1 更强传播器替换（Farneback → RAFT / XMem / SAM2）

* RAFT：更高质量光流，传播更稳，代价更高
* XMem：视频目标分割/VOS，更强遮挡鲁棒性（需权重与集成）
* SAM2/Video-SAM：交互式/半自动标注 + 时序一致性潜力

### 12.2 筛选策略增强

当前筛选使用相邻帧 warp+IoU，可进一步加入：

* 面积变化约束（mask area jump）
* 边界一致性（Boundary IoU）
* 形态学后处理（开闭运算、连通域过滤）
* 时序平滑（EMA/Kalman 对 mask 参数或概率图平滑）

### 12.3 训练目标增强

* 引入时序一致性 loss：`pred_t` warp 到 `t+1` 与 `pred_{t+1}` 约束
* 引入自训练/置信度加权训练（对伪标签噪声更稳）
* 引入轻量时序模块（TSM/ConvLSTM）提升视频一致性

### 12.4 实时化与工程部署

* FP16 / TensorRT / ONNX 加速
* 降分辨率推理 + ROI refinement
* 关键帧推理 + 中间帧传播（提升“实时体验”）

### 12.5 一键输出视频（run.sh 完整闭环）

工程化建议：

* 新增 `medvseg/engines/render_video.py`：

  * 读取 frames 与 `*_pred.png`
  * 阈值化、叠加、保存 overlay frames
  * 调用 ffmpeg 合成 mp4
* run.sh 执行结束时生成：

  * `outputs/videos/overlay.mp4`

---

## 13. 复现实验建议
* 服务器从github上拉代码时，采用
git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/JingchengLi2001/medvseg.git
### 13.1 最小复现

* 单视频（20s）+ 单帧标注（1 张）
* 完整跑通传播→筛选→训练→输出 overlay 视频

### 13.2 可靠性对比

* 单帧标注 vs 多帧标注（2/3/5 帧）
* Farneback vs RAFT（若引入）
* 筛选阈值 iou-th 变化对训练集质量与最终效果的影响

### 13.3 真实 GT 评估

* 额外手标若干帧作为 Test-only（不参与训练）
* 对比 Dice/IoU，并展示可视化 overlay

---

## 14. 许可与声明

该仓库用于毕业设计与研究复现。若集成第三方模型（XMem/RAFT 等），需遵守其原始许可证并在论文与仓库中标注来源。

```
::contentReference[oaicite:0]{index=0}
```