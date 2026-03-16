import yaml
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Any

from dataset import BrailleMultiLabelDataset, get_transforms
from model import BrailleDotNet
from metrics import compute_metrics

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_domain(file_name: str, domain_prefixes: dict[str, str]) -> str:
    for domain, prefix in domain_prefixes.items():
        if file_name.startswith(prefix):
            return domain
    return "Unknown"

@torch.no_grad()
def run_inference(
    model: torch.nn.Module, dataloader: DataLoader,
    device: torch.device, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_targets, all_preds = [], []
    
    for images_batch, targets_batch in dataloader:
        logits_batch = model(images_batch.to(device, non_blocking=True))
        all_preds.append((logits_batch > threshold).float().cpu().numpy())
        all_targets.append(targets_batch.numpy())
        
    return np.vstack(all_targets), np.vstack(all_preds)

def evaluate_split(
    split_name: str, split_dir: Path, model: torch.nn.Module, 
    config: dict[str, Any], domain_prefixes: dict[str, str], device: torch.device
) -> dict[tuple[str, str], Any] | None:
    if not split_dir.exists():
        return None

    data_cfg = config["data_processing"]
    dataset = BrailleMultiLabelDataset(
        data_dir=split_dir,
        transforms=get_transforms(data_cfg["mean"], data_cfg["std"], is_train=False),
        cache_in_ram=False,
    )
    dataloader = DataLoader(
        dataset, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=data_cfg["num_workers"],
    )

    y_true_batch, y_pred_batch = run_inference(model, dataloader, device, config["training"]["threshold"])
    image_domains = np.array([resolve_domain(p.name, domain_prefixes) for p in dataset.image_paths])

    results: dict[tuple[str, str], Any] = {("Metadata", "split"): split_name}
    results |= {("Overall", k): v for k, v in compute_metrics(y_true_batch, y_pred_batch).items()}

    for domain in domain_prefixes.keys():
        mask = image_domains == domain
        if not mask.any():
            continue
        results |= {(domain, k): v for k, v in compute_metrics(y_true_batch[mask], y_pred_batch[mask]).items()}

    return results

def export_report(all_results: list[dict[tuple[str, str], Any]], output_path: Path) -> None:
    df = pd.DataFrame(all_results)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.to_csv(output_path, index=False)

    console_df = df.set_index(("Metadata", "split"))
    console_df.index.name = "Split"
    
    print(f"\n{'=' * 80}\n{console_df.T.to_string()}\n{'=' * 80}")
    print(f"Report exported to: {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--cls-config", type=Path, default=Path("configs/classifier_config.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data_config.yaml"))
    args = parser.parse_args()

    config = load_yaml(args.cls_config)
    data_cfg = load_yaml(args.data_config)
    paths_cfg = config["paths"]

    domain_prefixes = {
        val["name"]: val["prefix"]
        for key, val in data_cfg["datasets"].items()
        if key != "merged" and "prefix" in val
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = Path(paths_cfg["project_dir"]).resolve() / paths_cfg["experiment_name"] / paths_cfg["best_model_name"]

    if not weights_path.exists():
        raise FileNotFoundError(f"Classifier weights not found: {weights_path}")

    print(f"Loading classifier from: {weights_path}")
    model = BrailleDotNet(num_classes=config["model"]["num_classes"])
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)

    dataset_dir = Path(paths_cfg["crops_dir"]).resolve()
    project_path = weights_path.parent
    project_path.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[tuple[str, str], Any]] = []
    
    for split in args.splits:
        if res := evaluate_split(split, dataset_dir / split, model, config, domain_prefixes, device):
            all_results.append(res)

    if all_results:
        export_report(all_results, project_path / "metrics_evaluation_summary.csv")

if __name__ == "__main__":
    main()