import argparse
import tempfile
import yaml
import pandas as pd
from pathlib import Path
from typing import Any
from ultralytics import YOLO

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_metrics(tag: str, metrics: Any) -> dict[tuple[str, str], float]:
    box = metrics.box
    
    p = float(box.mp)
    r = float(box.mr)
    f1 = 2 * (p * r) / (p + r + 1e-16)
    
def extract_metrics(tag: str, metrics: Any) -> dict[tuple[str, str], float]:
    box = metrics.box
    
    p = float(box.mp)
    r = float(box.mr)
    f1 = 2 * (p * r) / (p + r + 1e-16)
    
    cm = metrics.confusion_matrix.matrix
    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    
    return {
        (tag, "Precision"): round(p, 4),
        (tag, "Recall"):    round(r, 4),
        (tag, "F1"):        round(f1, 4),
        (tag, "mAP50"):     round(float(box.map50), 4),
        (tag, "mAP75"):     round(float(box.map75), 4),
        (tag, "mAP50-95"):  round(float(box.map), 4),
        (tag, "TP"):        int(tp),
        (tag, "FP"):        int(fp),
        (tag, "FN"):        int(fn),
    }

def export_report(all_results: list[dict[tuple[str, str], Any]], output_path: Path) -> None:
    df = pd.DataFrame(all_results)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.to_csv(output_path, index=False)
    
    console_df = df.set_index(("Metadata", "split"))
    console_df.index.name = "Split"
    
    print(f"\n{'=' * 80}\n EVALUATION METRICS SUMMARY\n{'=' * 80}")
    print(console_df.T.to_string())
    print(f"{'=' * 80}\n")

class YOLOEvaluator:
    def __init__(self, weights_path: Path, max_det: int, project_dir: Path, exp_name: str):
        self.model = YOLO(str(weights_path))
        self.max_det = max_det
        self.project_dir = project_dir
        self.exp_name = exp_name

    def evaluate_overall(self, yaml_path: Path, split: str) -> dict[tuple[str, str], float]:
        metrics = self.model.val(
            data=str(yaml_path),
            project=str(self.project_dir / self.exp_name),
            name=f"eval_{split}_overall",
            exist_ok=True,
            split=split,
            single_cls=True,
            max_det=self.max_det,
            verbose=False,
        )
        
        results = {("Metadata", "infer_ms"): round(metrics.speed["inference"], 2)}
        results |= extract_metrics("Overall", metrics)
        return results

    def evaluate_domain(
        self, domain_name: str, prefix: str, split: str, merged_yaml_path: Path, dataset_meta: dict[str, Any]
    ) -> dict[tuple[str, str], float] | None:
        img_dir = merged_yaml_path.parent / dataset_meta[split]
        domain_images = sorted(p for p in img_dir.glob(f"{prefix}*"))
        
        if not domain_images:
            return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            img_list_path = tmp_path / f"{domain_name}_images.txt"
            img_list_path.write_text("\n".join(str(p.resolve()) for p in domain_images))
            
            patched_yaml_path = tmp_path / f"{domain_name}_data.yaml"
            patched_yaml_path.write_text(yaml.dump({
                "nc": dataset_meta["nc"],
                "names": dataset_meta["names"],
                "train": str(img_list_path),
                "val": str(img_list_path),
                "test": str(img_list_path)
            }))
            
            metrics = self.model.val(
                data=str(patched_yaml_path),
                project=str(self.project_dir / self.exp_name),
                name=f"eval_{split}_{domain_name}",
                exist_ok=True,
                split=split,
                single_cls=True,
                max_det=self.max_det,
                verbose=False,
            )
            
            return extract_metrics(domain_name, metrics)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--det-config", type=Path, default=Path("configs/detector_config.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data_config.yaml"))
    args = parser.parse_args()

    det_cfg = load_yaml(args.det_config)
    data_cfg = load_yaml(args.data_config)

    proj_dir = Path(det_cfg["paths"]["project_dir"]).resolve()
    exp_name = det_cfg["paths"]["experiment_name"]
    weights_path = proj_dir / exp_name / "weights" / "best.pt"
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    domain_prefixes = {
        val["name"]: val["prefix"]
        for key, val in data_cfg["datasets"].items()
        if key != "merged" and "prefix" in val
    }

    merged_yaml_path = Path(det_cfg["paths"]["merged_yaml"]).resolve()
    dataset_meta = load_yaml(merged_yaml_path)

    evaluator = YOLOEvaluator(weights_path, det_cfg["training"]["max_det"], proj_dir, exp_name)
    all_results: list[dict[tuple[str, str], Any]] = []

    for split in args.splits:
        split_results: dict[tuple[str, str], Any] = {
            ("Metadata", "weights"): weights_path.name,
            ("Metadata", "split"): split,
        }

        split_results |= evaluator.evaluate_overall(merged_yaml_path, split)

        for domain_name, prefix in domain_prefixes.items():
            domain_metrics = evaluator.evaluate_domain(domain_name, prefix, split, merged_yaml_path, dataset_meta)
            if domain_metrics:
                split_results |= domain_metrics
            
        all_results.append(split_results)

    report_path = proj_dir / exp_name / "metrics_evaluation_summary.csv"
    export_report(all_results, report_path)

if __name__ == "__main__":
    main()