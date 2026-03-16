import sys
import argparse
import fiftyone as fo
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Explore MergedDataset-YOLO in FiftyOne.")
    parser.add_argument("--cache", action="store_true", help="Use FiftyOne database caching for fast loading")
    args = parser.parse_args()

    ds_dir = Path(__file__).resolve().parents[2] / "datasets" / "MergedDataset-YOLO"
    yaml_file = ds_dir / "data.yaml"
    
    if not yaml_file.exists():
        sys.exit(f"Error: Dataset not found at {ds_dir}\nPlease download it")

    ds_name = "MergedDataset-YOLO"
    
    if args.cache and fo.dataset_exists(ds_name):
        print("Found cached dataset. Loading directly from database...")
        dataset = fo.load_dataset(ds_name)
    else:
        if fo.dataset_exists(ds_name):
            fo.delete_dataset(ds_name)

        dataset = fo.Dataset(ds_name)
        dataset.persistent = args.cache
        
        for split in ["train", "val", "test"]:
            if (ds_dir / "images" / split).exists():
                print(f"Ingesting {split}...")
                dataset.add_dir(
                    dataset_dir=str(ds_dir),
                    yaml_path=str(yaml_file),
                    dataset_type=fo.types.YOLOv5Dataset,
                    split=split,
                    tags=[split]
                )

    print(f"\nDataset ready: {len(dataset)} samples.\nLaunching FiftyOne... (Press Ctrl+C to exit)")
    fo.launch_app(dataset).wait()

if __name__ == "__main__":
    main()