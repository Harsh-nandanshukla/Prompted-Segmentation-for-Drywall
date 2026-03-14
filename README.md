# Prompted Segmentation for Drywall QA

Zero-shot text-conditioned segmentation using **Grounded SAM** (GroundingDINO + SAM ViT-B).

---

## Approach

Given an image and a natural-language prompt, the pipeline produces a binary mask:

| Prompt | Dataset |
|--------|---------|
| `segment taping area` | Drywall-Join-Detect-1 |
| `segment crack` | cracks-1 |

**Pipeline:**
1. GroundingDINO detects bounding boxes from the text prompt (zero-shot)
2. SAM (ViT-B) converts each box into a precise binary mask
3. All masks are union-merged into one prediction per image

---

## Models

| Model | Variant | Checkpoint |
|-------|---------|------------|
| GroundingDINO | SwinT-OGC | `groundingdino_swint_ogc.pth` |
| SAM | ViT-B | `sam_vit_b_01ec64.pth` |

---

## Project Structure

```
segmentation_drywall/
├── cracks-1/                        # Dataset 2 (not tracked in git)
│   ├── train/  (images + _annotations.coco.json)
│   ├── valid/
│   └── test/
├── Drywall-Join-Detect-1/           # Dataset 1 (not tracked in git)
│   ├── train/
│   └── valid/
├── GroundingDINO/                   # Cloned repo (not tracked in git)
│   ├── groundingdino/
│   └── weights/
│       └── groundingdino_swint_ogc.pth
├── weights/                         # SAM weights (not tracked in git)
│   └── sam_vit_b_01ec64.pth
├── outputs/                         # Prediction masks (not tracked in git)
├── results/
│   ├── metrics.json
│   ├── metrics.md
│   └── visuals/
│       ├── drywall_examples.png
│       └── cracks_examples.png
├── inference.py
├── evaluate.py
├── visualize.py
├── ms_deform_attn.py                # Patched file for Windows CUDA fix
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone GroundingDINO
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install -e ./GroundingDINO --no-build-isolation
```

### 2. Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install segment-anything roboflow opencv-python-headless matplotlib
pip install Pillow pycocotools supervision tqdm scikit-learn
```

### 3. Download weights

**SAM ViT-B:**
```powershell
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -OutFile "weights/sam_vit_b_01ec64.pth"
```

**GroundingDINO:**
```powershell
Invoke-WebRequest -Uri "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" -OutFile "GroundingDINO/weights/groundingdino_swint_ogc.pth"
```

### 4. Windows CUDA fix

Replace the file at:
```
GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py
```
with the patched `ms_deform_attn.py` provided in this repo. This replaces the uncompilable CUDA C++ extension with a pure PyTorch fallback using `F.grid_sample`.

### 5. Download datasets

From [Roboflow Universe](https://universe.roboflow.com) in **COCO JSON** format:
- [Drywall-Join-Detect](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect)
- [Cracks](https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36)

Place them as `Drywall-Join-Detect-1/` and `cracks-1/` in the project root.

---

## Run

```bash
conda activate gsam

# Step 1 — generate prediction masks (~3 hrs on GPU)
python inference.py

# Step 2 — compute mIoU & Dice
python evaluate.py

# Step 3 — generate visual examples
python visualize.py
```

---

## Results

| Dataset | Prompt | Split | Images | mIoU | Dice |
|---------|--------|-------|--------|------|------|
| Drywall-Join-Detect-1 | segment_taping_area | valid | 250/250 | 0.1236 | 0.1975 |
| cracks-1 | segment_crack | test | 4/4 | 0.1394 | 0.2280 |

> Zero-shot baseline — no fine-tuning applied. Supervised models typically achieve 0.50–0.75 mIoU on these tasks.

---

## Output Mask Format

- Single-channel PNG, pixel values `{0, 255}`
- Same spatial size as source image
- Filename: `{image_id}__{prompt_tag}.png`
  - e.g. `IMG_0042__segment_crack.png`
  - e.g. `frame_001__segment_taping_area.png`

---

## Seeds

All random seeds fixed to **42**:
```python
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

---

## Notes

- Drywall dataset exports contain bounding boxes only (no polygon masks). GT masks are constructed from bbox regions.
- Images larger than 1024px are resized proportionally at inference to avoid OOM errors.
- The GroundingDINO CUDA extension (`_C`) does not compile on Windows — use the provided patched `ms_deform_attn.py`.
