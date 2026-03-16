import sys
import cv2
import yaml
import torch
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Any
from torchvision.ops import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.pipeline_model import EndToEndPipeline

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)

def match_predictions(
        gt_boxes: list[tuple], gt_chars: list[str],
        pred_boxes: list[tuple], pred_chars: list[str],
        iou_thresh: float = 0.5
) -> tuple[int, int, int]:
    if not gt_boxes: return 0, len(pred_boxes), 0
    if not pred_boxes: return 0, 0, len(gt_boxes)

    ious = box_iou(torch.tensor(pred_boxes, dtype=torch.float32),
                   torch.tensor(gt_boxes, dtype=torch.float32))

    tp, matched_gt = 0, set()

    for p_idx in torch.argsort(ious.max(dim=1).values, descending=True).tolist():
        row = ious[p_idx].clone()
        for m_idx in matched_gt:
            row[m_idx] = -1.0

        best_iou, g_idx = row.max(0)
        if best_iou.item() >= iou_thresh:
            g_idx_val = g_idx.item()
            matched_gt.add(g_idx_val)

            if str(pred_chars[p_idx]) == str(gt_chars[g_idx_val]):
                tp += 1

    return tp, len(pred_boxes) - tp, len(gt_boxes) - tp

def compute_prf1(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "Recall": round(r, 4),
        "Precision": round(p, 4),
        "F1_Score": round(f1, 4),
        "Strict_Acc": round(acc, 4)
    }

def load_gt(img_path: Path, img_shape: tuple[int, int], class_names: list[str]) -> tuple[list[tuple], list[str]]:
    lbl_path = img_path.parents[2] / "labels" / img_path.parent.name / f"{img_path.stem}.txt"
    if not lbl_path.exists():
        return [], []

    h, w = img_shape
    boxes, chars = [], []
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        if not (parts := line.split()): continue

        cls_id, xc, yc, bw, bh = map(float, parts)
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)

        boxes.append((x1, y1, x2, y2))
        chars.append(class_names[int(cls_id)])

    return boxes, chars

def evaluate_split(
        pipeline: EndToEndPipeline, img_paths: list[Path],
        class_names: list[str], iou_thresh: float, domain_prefixes: dict[str, str]
) -> dict[tuple[str, str], Any]:

    stats = {"Overall": {"tp": 0, "fp": 0, "fn": 0}} | {
        k: {"tp": 0, "fp": 0, "fn": 0} for k in domain_prefixes
    }

    inference_times = []

    for path in tqdm(img_paths, desc="Evaluating", leave=False):
        img = cv2.imread(str(path))
        if img is None: continue

        gt_boxes, gt_chars = load_gt(path, img.shape[:2], class_names)
        if not gt_boxes: continue

        start_inf = time.perf_counter()
        preds = pipeline.process_image(img, translate=False)
        inference_times.append(time.perf_counter() - start_inf)

        tp, fp, fn = match_predictions(
            gt_boxes, gt_chars,
            [p["bbox"] for p in preds], [p["dots"] for p in preds],
            iou_thresh
        )

        for tag in ["Overall"] + [k for k, pfx in domain_prefixes.items() if path.name.startswith(pfx)]:
            stats[tag]["tp"] += tp
            stats[tag]["fp"] += fp
            stats[tag]["fn"] += fn

    avg_ms = (sum(inference_times) / len(inference_times)) * 1000 if inference_times else 0

    results = {
        (dom, met): val for dom, counts in stats.items()
        for met, val in compute_prf1(**counts).items()
    }

    results[("Overall", "Inf_ms")] = round(avg_ms, 2)
    results[("Overall", "FPS")] = round(1000 / avg_ms, 2) if avg_ms > 0 else 0

    return results

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline_config.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data_config.yaml"))
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = load_yaml(args.data_config)

    merged_yaml = Path(cfg["paths"]["merged_yaml"])
    class_names = load_yaml(merged_yaml)["names"]

    domain_prefixes = {v["name"]: v["prefix"] for v in data_cfg["datasets"].values() if "prefix" in v}

    exp_dir = Path(cfg["paths"]["output_dir"]) / cfg["paths"].get("experiment_name", "eval_run")
    exp_dir.mkdir(parents=True, exist_ok=True)

    pipeline = EndToEndPipeline(str(args.config))
    all_results = []

    for split in args.splits:
        img_paths = sorted((merged_yaml.parent / "images" / split).glob("*.jpg"))
        if not img_paths: continue

        res = evaluate_split(pipeline, img_paths, class_names, args.iou_thresh, domain_prefixes)
        res[("Metadata", "split")] = split
        all_results.append(res)

    df = pd.DataFrame(all_results)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.to_csv(exp_dir / "pipeline_metrics_summary.csv", index=False)

    print(f"\n{'=' * 80}\n PIPELINE FINAL REPORT\n{'=' * 80}")
    print(df.set_index(("Metadata", "split")).T.to_string())
    print(f"{'=' * 80}\n")

if __name__ == "__main__":
    main()