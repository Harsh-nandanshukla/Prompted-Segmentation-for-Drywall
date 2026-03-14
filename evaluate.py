"""
evaluate.py  —  Compute mIoU & Dice for both prompts
Reads GT from _annotations.coco.json, pred from outputs/

Usage:
    python evaluate.py
"""

import json, re
import numpy as np
from pathlib import Path
from pycocotools import mask as coco_mask_util
from pycocotools.coco import COCO
from tqdm import tqdm
import cv2

ROOT       = Path(__file__).parent
OUTPUT_DIR = ROOT / "outputs"
RESULTS    = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

DATASETS = [
    {
        "name":   "drywall",
        "root":   ROOT / "Drywall-Join-Detect-1",
        "splits": ["valid"],          # evaluate on val (test split absent)
        "tag":    "segment_taping_area",
    },
    {
        "name":   "cracks",
        "root":   ROOT / "cracks-1",
        "splits": ["test"],           # prefer test; falls back to valid
        "tag":    "segment_crack",
    },
]


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)


# def coco_anns_to_mask(coco: COCO, img_info: dict) -> np.ndarray:
#     """Merge all annotations for one image into a single binary GT mask."""
#     h, w   = img_info["height"], img_info["width"]
#     ann_ids = coco.getAnnIds(imgIds=img_info["id"])
#     anns    = coco.loadAnns(ann_ids)
#     gt      = np.zeros((h, w), dtype=np.uint8)
#     for ann in anns:
#         if "segmentation" not in ann:
#             continue
#         seg = ann["segmentation"]
#         if isinstance(seg, list):                 # polygon
#             for poly in seg:
#                 pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
#                 cv2.fillPoly(gt, [pts], 1)
#         elif isinstance(seg, dict):               # RLE
#             rle_mask = coco_mask_util.decode(seg)
#             gt = np.logical_or(gt, rle_mask).astype(np.uint8)
#     return gt
def coco_anns_to_mask(coco: COCO, img_info: dict) -> np.ndarray:
    """Merge all annotations into a single binary GT mask.
    Uses segmentation polygons if available, falls back to bbox."""
    h, w    = img_info["height"], img_info["width"]
    ann_ids = coco.getAnnIds(imgIds=img_info["id"])
    anns    = coco.loadAnns(ann_ids)
    gt      = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        seg = ann.get("segmentation", [])
        if isinstance(seg, list) and len(seg) > 0:
            # polygon
            for poly in seg:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(gt, [pts], 1)
        elif isinstance(seg, dict):
            # RLE
            rle_mask = coco_mask_util.decode(seg)
            gt = np.logical_or(gt, rle_mask).astype(np.uint8)
        else:
            # fallback: use bounding box as GT region
            if "bbox" in ann:
                x, y, bw, bh = [int(v) for v in ann["bbox"]]
                gt[y:y+bh, x:x+bw] = 1
    return gt


def iou_dice(pred: np.ndarray, gt: np.ndarray):
    pred_b = (pred > 127).astype(bool)
    gt_b   = (gt   > 0  ).astype(bool)
    inter  = (pred_b & gt_b).sum()
    union  = (pred_b | gt_b).sum()
    iou    = inter / union if union > 0 else float("nan")
    dice   = 2 * inter / (pred_b.sum() + gt_b.sum()) if (pred_b.sum() + gt_b.sum()) > 0 else float("nan")
    return iou, dice


def evaluate_split(ds: dict, split: str):
    ann_file = ds["root"] / split / "_annotations.coco.json"
    img_dir  = ds["root"] / split 
    tag      = ds["tag"]

    if not ann_file.exists():
        print(f"  [skip] {ann_file} not found")
        return None

    coco   = COCO(str(ann_file))
    img_ids = coco.getImgIds()

    ious, dices = [], []
    missing = 0

    for img_id in tqdm(img_ids, desc=f"{ds['name']}/{split}"):
        img_info = coco.loadImgs(img_id)[0]
        stem     = safe_stem(img_info["file_name"])
        pred_path = OUTPUT_DIR / f"{stem}__{tag}.png"

        if not pred_path.exists():
            missing += 1
            continue

        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        gt   = coco_anns_to_mask(coco, img_info)

        # Resize pred to gt size if needed
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        iou, dice = iou_dice(pred, gt)
        ious.append(iou)
        dices.append(dice)

    valid_ious  = [v for v in ious  if not np.isnan(v)]
    valid_dices = [v for v in dices if not np.isnan(v)]

    result = {
        "dataset":  ds["name"],
        "split":    split,
        "tag":      tag,
        "n_images": len(img_ids),
        "n_scored": len(valid_ious),
        "missing":  missing,
        "mIoU":     float(np.mean(valid_ious))  if valid_ious  else 0.0,
        "Dice":     float(np.mean(valid_dices)) if valid_dices else 0.0,
    }
    return result


def run():
    all_results = []

    for ds in DATASETS:
        for split in ds["splits"]:
            res = evaluate_split(ds, split)
            if res:
                all_results.append(res)
                print(f"\n{'='*50}")
                print(f"  Dataset : {res['dataset']}  ({res['tag']})")
                print(f"  Split   : {res['split']}")
                print(f"  Images  : {res['n_scored']} / {res['n_images']}  (missing: {res['missing']})")
                print(f"  mIoU    : {res['mIoU']:.4f}")
                print(f"  Dice    : {res['Dice']:.4f}")

    # Save JSON
    out_json = RESULTS / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save markdown table
    out_md = RESULTS / "metrics.md"
    with open(out_md, "w") as f:
        f.write("# Evaluation Results\n\n")
        f.write("| Dataset | Prompt | Split | Images | mIoU | Dice |\n")
        f.write("|---------|--------|-------|--------|------|------|\n")
        for r in all_results:
            f.write(f"| {r['dataset']} | {r['tag']} | {r['split']} "
                    f"| {r['n_scored']}/{r['n_images']} "
                    f"| {r['mIoU']:.4f} | {r['Dice']:.4f} |\n")

    print(f"\nResults saved → {RESULTS}/")


if __name__ == "__main__":
    run()
