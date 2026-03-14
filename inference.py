"""
inference.py  —  Grounded SAM prediction mask generator
Produces binary PNG masks (0/255) for:
  - "segment taping area"  (Drywall-Join-Detect-1)
  - "segment crack"        (cracks-1)

Usage:
    python inference.py
"""

import os, sys, random, json, re
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
GDINO_REPO    = ROOT / "GroundingDINO"
SAM_CKPT      = ROOT / "weights" / "sam_vit_b_01ec64.pth"          # root/weights/
GDINO_CKPT    = GDINO_REPO / "weights" / "groundingdino_swint_ogc.pth"  # GroundingDINO/weights/
GDINO_CFG     = GDINO_REPO / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
OUTPUT_DIR    = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Dataset configs ───────────────────────────────────────────────────────────
DATASETS = [
    {
        "name":    "drywall",
        "root":    ROOT / "Drywall-Join-Detect-1",
        "splits":  ["train", "valid"],
        "prompt":  "taping area . joint tape . drywall seam",
        "tag":     "segment_taping_area",
    },
    {
        "name":    "cracks",
        "root":    ROOT / "cracks-1",
        "splits":  ["train", "valid", "test"],
        "prompt":  "crack . wall crack . fracture",
        "tag":     "segment_crack",
    },
]

# ── GroundingDINO threshold ────────────────────────────────────────────────────
BOX_THRESHOLD  = 0.25
TEXT_THRESHOLD = 0.25

# ── Load models ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(GDINO_REPO))

def load_models():
    # GroundingDINO
    from groundingdino.util.inference import load_model
    gdino = load_model(str(GDINO_CFG), str(GDINO_CKPT))
    gdino.eval()

    # SAM
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device)
    predictor = SamPredictor(sam)

    print(f"Models loaded  |  device: {device}")
    return gdino, predictor

# ── Inference helpers ─────────────────────────────────────────────────────────
def predict_masks(gdino, predictor, image_path: Path, prompt: str):
    """
    Returns list of binary uint8 masks (H x W, values 0/255).
    One mask per detected box; if no box found returns [zeros mask].
    """
    from groundingdino.util.inference import load_image, predict

    # image_source, image_tensor = load_image(str(image_path))
    # h, w = image_source.shape[:2]
    image_source, image_tensor = load_image(str(image_path))
    h, w = image_source.shape[:2]

    # Resize large images to avoid OOM (preserve aspect ratio)
    MAX_DIM = 1024
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image_source = cv2.resize(image_source, (new_w, new_h))
        # from groundingdino.util.inference import load_image as _li
        import torchvision.transforms as T
        image_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])(Image.fromarray(image_source))
        h, w = new_h, new_w

    # 1. GroundingDINO → boxes
    boxes, logits, phrases = predict(
        model=gdino,
        image=image_tensor,
        caption=prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    if boxes is None or len(boxes) == 0:
        return [np.zeros((h, w), dtype=np.uint8)]

    # Convert cx,cy,w,h (normalised) → xyxy pixels
    boxes_xyxy = boxes.clone()
    boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
    boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
    boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
    boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h

    # 2. SAM → masks
    predictor.set_image(image_source)
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes_xyxy, image_source.shape[:2]
    ).to(predictor.device)

    masks_out, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # Union of all masks into one binary mask
    union = np.zeros((h, w), dtype=np.uint8)
    for m in masks_out:
        union = np.logical_or(union, m[0].cpu().numpy()).astype(np.uint8)

    return [union * 255]


def safe_stem(path: Path) -> str:
    """Filename-safe image id."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", path.stem)


# ── Main loop ─────────────────────────────────────────────────────────────────
def run():
    gdino, predictor = load_models()

    for ds in DATASETS:
        ds_root = ds["root"]
        prompt  = ds["prompt"]
        tag     = ds["tag"]

        for split in ds["splits"]:
            split_dir = ds_root / split 
            if not split_dir.exists():
                print(f"  [skip] {split_dir} not found")
                continue

            image_paths = sorted(
                list(split_dir.glob("*.jpg")) +
                list(split_dir.glob("*.jpeg")) +
                list(split_dir.glob("*.png"))
            )
            image_paths = [p for p in image_paths if "_annotations" not in p.name]
            print(f"\n[{ds['name']}] {split}  —  {len(image_paths)} images")

            for img_path in tqdm(image_paths):
                img_id  = safe_stem(img_path)
                out_name = f"{img_id}__{tag}.png"
                out_path = OUTPUT_DIR / out_name

                if out_path.exists():
                    continue  # resume-friendly

                masks = predict_masks(gdino, predictor, img_path, prompt)
                # Save the union mask
                Image.fromarray(masks[0]).save(out_path)

    print(f"\nDone. Masks saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
