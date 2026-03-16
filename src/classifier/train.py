import yaml
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any
from torch.amp import autocast, GradScaler

from dataset import BrailleMultiLabelDataset, get_transforms
from model import BrailleDotNet
from metrics import compute_metrics

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(
    model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
    criterion: nn.Module, scaler: GradScaler, scheduler: Any,
    device: torch.device, epoch: int, total_epochs: int, threshold: float
) -> tuple[float, dict[str, float]]:
    model.train()
    running_loss = 0.0
    all_targets, all_preds = [], []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:03d}/{total_epochs} [Train]", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        all_targets.append(targets.cpu().numpy())
        all_preds.append((logits.detach() > threshold).float().cpu().numpy())

        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)
    return running_loss / len(dataloader.dataset), compute_metrics(y_true, y_pred)

@torch.no_grad()
def validate_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
    device: torch.device, threshold: float
) -> tuple[float, dict[str, float]]:
    model.eval()
    running_loss = 0.0
    all_targets, all_preds = [], []

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type=device.type):
            logits = model(images)
            loss = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)
        all_targets.append(targets.cpu().numpy())
        all_preds.append((logits > threshold).float().cpu().numpy())

    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)
    return running_loss / len(dataloader.dataset), compute_metrics(y_true, y_pred)

def main() -> None:
    config = load_yaml(Path("configs/classifier_config.yaml"))
    paths_cfg = config["paths"]
    train_cfg = config["training"]
    data_cfg = config["data_processing"]

    set_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crops_dir = Path(paths_cfg["crops_dir"]).resolve()

    train_ds = BrailleMultiLabelDataset(
        crops_dir / "train",
        get_transforms(data_cfg["mean"], data_cfg["std"], is_train=True),
        is_train=True, cache_in_ram=True,
    )
    val_ds = BrailleMultiLabelDataset(
        crops_dir / "val",
        get_transforms(data_cfg["mean"], data_cfg["std"], is_train=False),
        is_train=False, cache_in_ram=True,
    )

    loader_kwargs = {"batch_size": train_cfg["batch_size"], "num_workers": train_cfg["num_workers"], "pin_memory": True}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = BrailleDotNet(num_classes=config["model"]["num_classes"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=float(train_cfg["weight_decay"]))
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=train_cfg["learning_rate"],
        steps_per_epoch=len(train_loader), epochs=train_cfg["epochs"],
    )
    scaler = GradScaler(device.type)

    weights_path = Path(paths_cfg["project_dir"]) / paths_cfg["experiment_name"] / paths_cfg["best_model_name"]
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    total_epochs = train_cfg["epochs"]
    threshold = train_cfg["threshold"]
    best_val_char_acc = 0.0
    epochs_no_improve = 0
    log_rows: list[dict[str, Any]] = []

    print(f"Starting Training: {paths_cfg['experiment_name']} on {device}")

    for epoch in range(1, total_epochs + 1):
        train_loss, train_m = train_epoch(
            model, train_loader, optimizer, criterion, scaler, scheduler, device, epoch, total_epochs, threshold
        )
        val_loss, val_m = validate_epoch(model, val_loader, criterion, device, threshold)

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
        })

        print(
            f"Ep {epoch:03d}/{total_epochs} | Loss: {train_loss:.4f} / {val_loss:.4f}\n"
            f"  Char — Acc: {train_m['CharAccuracy']:.1%} / {val_m['CharAccuracy']:.1%} | "
            f"P: {train_m['CharPrecision']:.1%} / {val_m['CharPrecision']:.1%} | "
            f"R: {train_m['CharRecall']:.1%} / {val_m['CharRecall']:.1%} | "
            f"F1: {train_m['CharF1']:.1%} / {val_m['CharF1']:.1%}\n"
            f"  Dot  — Acc: {train_m['DotAccuracy']:.1%} / {val_m['DotAccuracy']:.1%}"
        )

        if val_m["CharAccuracy"] > best_val_char_acc:
            best_val_char_acc = val_m["CharAccuracy"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), weights_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= train_cfg["patience"]:
                print(f"Early Stopping triggered at epoch {epoch}.")
                break

    pd.DataFrame(log_rows).to_csv(weights_path.parent / "train_log.csv", index=False)
    print(f"Best val CharAccuracy: {best_val_char_acc:.2%}")

if __name__ == "__main__":
    main()