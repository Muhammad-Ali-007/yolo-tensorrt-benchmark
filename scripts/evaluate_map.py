#!/usr/bin/env python3
"""mAP@0.5:0.95 evaluation per engine, against COCO val2017 ground truth.

This is what tells you what each precision level actually costs — FP16 is
usually near-free, INT8 without good calibration can quietly cost several
mAP points, and this is the script that catches that instead of assuming.

Usage:
    python scripts/evaluate_map.py --config configs/models.yaml --data data/val2017
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.trt_infer import TensorRTInfer

# Standard COCO 80-class index -> category_id mapping (skips the 11 unused ids)
COCO80_TO_COCO91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
    63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85,
    86, 87, 88, 89, 90,
]


def preprocess(image_bgr, input_size):
    h0, w0 = image_bgr.shape[:2]
    img = cv2.resize(image_bgr, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))[None]
    return img, (w0 / input_size, h0 / input_size)


def postprocess_to_coco_dets(raw_output, scale_xy, image_id,
                             conf_thresh=0.25, iou_thresh=0.7):
    """Decode a YOLO detection head output into COCO-format detection dicts.

    Ultralytics YOLOv8/v11 ONNX exports emit [1, 4+nc, 8400] — *channels
    first*, i.e. the transpose of "one row per candidate". Verify with
    `raw_output.shape` if you swap in a model from another family; the
    orientation check below handles both layouts.

    NMS is not optional here. The raw head emits ~8400 candidates and a
    single object typically survives conf-thresholding as dozens of
    near-duplicate boxes. COCOeval matches one of them as a true positive
    and counts the rest as false positives, which quietly collapses
    precision — the failure looks like "INT8 hurt accuracy" when in fact
    the decode was wrong.
    """
    preds = np.squeeze(raw_output)
    if preds.ndim != 2:
        raise ValueError(
            f"expected a 2D detection head after squeeze, got {raw_output.shape}"
        )
    # (4+nc, num_anchors) -> (num_anchors, 4+nc)
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T

    boxes_cxcywh = preds[:, :4]
    class_scores = preds[:, 4:]
    cls_ids = np.argmax(class_scores, axis=1)
    confs = class_scores[np.arange(class_scores.shape[0]), cls_ids]

    keep = confs >= conf_thresh
    if not np.any(keep):
        return []
    boxes_cxcywh, cls_ids, confs = boxes_cxcywh[keep], cls_ids[keep], confs[keep]

    # cxcywh (model input space) -> xywh top-left, rescaled to original image
    sx, sy = scale_xy
    cx, cy, w, h = boxes_cxcywh.T
    x = (cx - w / 2) * sx
    y = (cy - h / 2) * sy
    w = w * sx
    h = h * sy
    boxes_xywh = np.stack([x, y, w, h], axis=1)

    # Class-aware NMS: boxes of different classes must not suppress each other.
    kept_idx = cv2.dnn.NMSBoxesBatched(
        boxes_xywh.tolist(), confs.astype(np.float32).tolist(),
        cls_ids.astype(np.int32).tolist(), conf_thresh, iou_thresh,
    )
    if len(kept_idx) == 0:
        return []

    return [
        {
            "image_id": image_id,
            "category_id": COCO80_TO_COCO91[int(cls_ids[i])],
            "bbox": [float(v) for v in boxes_xywh[i]],
            "score": float(confs[i]),
        }
        for i in np.array(kept_idx).flatten()
    ]


def evaluate_engine(engine_path, coco, img_dir, input_size, limit=None):
    runner = TensorRTInfer(engine_path)
    img_ids = coco.getImgIds()
    if limit:
        img_ids = img_ids[:limit]

    all_dets = []
    for i, img_id in enumerate(img_ids):
        info = coco.loadImgs(img_id)[0]
        img_path = Path(img_dir) / info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        inp, scale_xy = preprocess(image, input_size)
        outputs = runner.infer(inp)
        all_dets.extend(postprocess_to_coco_dets(outputs[0], scale_xy, img_id))

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(img_ids)} images...")

    if not all_dets:
        print("  WARNING: no detections above threshold — mAP will be 0. "
              "Check postprocess_to_coco_dets() matches your model's output layout.")
        return {"mAP@0.5:0.95": 0.0, "mAP@0.5": 0.0}

    coco_dt = coco.loadRes(all_dets)
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {"mAP@0.5:0.95": coco_eval.stats[0], "mAP@0.5": coco_eval.stats[1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="dir containing val2017 images")
    ap.add_argument("--annotations", default="data/annotations/instances_val2017.json")
    ap.add_argument("--limit", type=int, default=200,
                     help="images to evaluate per engine (200 is fast+indicative; "
                          "use the full 5000 for a publishable number)")
    ap.add_argument("--out", default="results/accuracy_results.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    input_size = cfg.get("input_size", 640)
    coco = COCO(args.annotations)

    rows = []
    for model_cfg in cfg["models"]:
        name = model_cfg["name"]
        for precision, engine_path in model_cfg["engines"].items():
            if not Path(engine_path).exists():
                print(f"[{name}] {precision}: skipping — {engine_path} not built yet")
                continue
            print(f"[{name}] {precision}: evaluating on {args.limit} images...")
            metrics = evaluate_engine(engine_path, coco, args.data, input_size, args.limit)
            rows.append({"model": name, "precision": precision, **metrics})

    if not rows:
        print("No engines found to evaluate — run build_engine.py first.")
        return

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
