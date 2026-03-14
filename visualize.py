"""
visualize.py  —  Generate orig | GT | pred comparison figures
Saves 4 examples per dataset to results/visuals/

Usage:
    python visualize.py
"""

import json, re, random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_util

SEED = 42
random.seed(SEED)

ROOT       = Path(__file__).parent
OUTPUT_DIR = ROOT / "outputs"
RESULTS    = ROOT / "results" / "visuals"
RESULTS.mkdir(parents=True, exist_ok=True)

DATASETS = [
    {
        "name":   "drywall",
        "root":   ROOT / "Drywall-Join-Detect-1",
        "split":  "valid",
        "tag":    "segment_taping_area",
        "color":  (0.2, 0.6, 1.0),   # blue overlay
    },
    {
        "name":   "cracks",
        "root":   ROOT / "cracks-1",
        "split":  "test",
        "tag":    "segment_crack",
        "color":  (1.0, 0.3, 0.2),   # red overlay
    },
]

N_EXAMPLES = 4


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)


def coco_anns_to_mask(coco, img_info):
    h, w    = img_info["height"], img_info["width"]
    ann_ids = coco.getAnnIds(imgIds=img_info["id"])
    anns    = coco.loadAnns(ann_ids)
    gt      = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        seg = ann.get("segmentation", [])
        if isinstance(seg, list) and len(seg) > 0:
            for poly in seg:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(gt, [pts], 1)
        elif isinstance(seg, dict):
            rle_mask = coco_mask_util.decode(seg)
            gt = np.logical_or(gt, rle_mask).astype(np.uint8)
        else:
            if "bbox" in ann:
                x, y, bw, bh = [int(v) for v in ann["bbox"]]
                gt[y:y+bh, x:x+bw] = 1
    return gt


def overlay(image_rgb: np.ndarray, mask: np.ndarray, color, alpha=0.45):
    """Blend a boolean mask onto an RGB image."""
    out = image_rgb.copy().astype(float) / 255.0
    for c, val in enumerate(color):
        out[:, :, c] = np.where(mask > 0,
                                out[:, :, c] * (1 - alpha) + val * alpha,
                                out[:, :, c])
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def iou_dice(pred, gt):
    p = (pred > 127).astype(bool)
    g = (gt > 0).astype(bool)
    inter = (p & g).sum()
    union = (p | g).sum()
    iou  = inter / union if union > 0 else float("nan")
    dice = 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) > 0 else float("nan")
    return iou, dice


def visualize_dataset(ds: dict):
    ann_file = ds["root"] / ds["split"] / "_annotations.coco.json"
    img_dir  = ds["root"] / ds["split"] 
    tag      = ds["tag"]
    color    = ds["color"]

    if not ann_file.exists():
        # fallback to valid
        ann_file = ds["root"] / "valid" / "_annotations.coco.json"
        img_dir  = ds["root"] / "valid" / "images"

    coco    = COCO(str(ann_file))
    img_ids = coco.getImgIds()

    # Pick images that have a prediction mask
    candidates = []
    for img_id in img_ids:
        img_info  = coco.loadImgs(img_id)[0]
        stem      = safe_stem(img_info["file_name"])
        pred_path = OUTPUT_DIR / f"{stem}__{tag}.png"
        if pred_path.exists():
            candidates.append((img_info, pred_path))

    if not candidates:
        print(f"  [skip] No predictions found for {ds['name']}")
        return

    samples = random.sample(candidates, min(N_EXAMPLES, len(candidates)))

    fig, axes = plt.subplots(len(samples), 3,
                             figsize=(12, 4 * len(samples)),
                             tight_layout=True)
    if len(samples) == 1:
        axes = [axes]

    col_titles = ["Original", "Ground Truth", "Prediction"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=13, fontweight="bold", pad=8)

    for row, (img_info, pred_path) in enumerate(samples):
        # Load image
        img_file = img_dir / img_info["file_name"]
        if not img_file.exists():
            img_file = img_dir / Path(img_info["file_name"]).name
        img_bgr  = cv2.imread(str(img_file))
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # GT mask
        gt   = coco_anns_to_mask(coco, img_info)

        # Pred mask
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        iou, dice = iou_dice(pred, gt)

        # Column 0: original
        axes[row][0].imshow(img_rgb)
        axes[row][0].axis("off")

        # Column 1: GT overlay
        gt_overlay = overlay(img_rgb, gt, color)
        axes[row][1].imshow(gt_overlay)
        axes[row][1].axis("off")

        # Column 2: Pred overlay + metrics
        pred_overlay = overlay(img_rgb, pred, color)
        axes[row][2].imshow(pred_overlay)
        axes[row][2].set_xlabel(
            f"IoU={iou:.3f}  Dice={dice:.3f}",
            fontsize=10, color="#333333"
        )
        axes[row][2].axis("off")

    # Legend patch
    patch = mpatches.Patch(color=color, label=tag.replace("_", " "), alpha=0.7)
    fig.legend(handles=[patch], loc="lower center",
               fontsize=10, ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.01))

    out_path = RESULTS / f"{ds['name']}_examples.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def run():
    for ds in DATASETS:
        print(f"\n[{ds['name']}] generating visuals...")
        visualize_dataset(ds)
    print("\nDone.")


if __name__ == "__main__":
    run()
