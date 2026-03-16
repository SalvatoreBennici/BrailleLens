import yaml
from pathlib import Path
from typing import Any, Dict
from ultralytics import YOLO

def load_config(config_path: str = "configs/detector_config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def train_model(config: Dict[str, Any]) -> None:
    model_cfg = config["model"]
    train_cfg = config["training"]
    hardware_cfg = config["hardware"]
    loss_cfg = config["loss"]
    aug_cfg = config["augmentations"]
    paths_cfg = config["paths"]

    project_dir = Path(paths_cfg["project_dir"]).resolve()
    merged_yaml = Path(paths_cfg["merged_yaml"]).resolve()

    if not merged_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML missing: {merged_yaml}")

    model_weights = model_cfg.pop("weights")
    
    print(f"Initializing YOLO model: {model_weights}")
    model = YOLO(model_weights)

    print(f"Starting training on: {merged_yaml}")
    model.train(
        data=str(merged_yaml),
        project=str(project_dir),
        name=paths_cfg["experiment_name"],
        **model_cfg,    # imgsz, single_cls, etc.
        **hardware_cfg, # device, workers, seed, deterministic, etc.
        **train_cfg,    # epochs, batch, optimizer, etc.
        **loss_cfg,     # box, cls. etc.
        **aug_cfg       # augment, mosaic, flipud, etc.
    )

def main() -> None:
    config = load_config()
    train_model(config)

if __name__ == "__main__":
    main()