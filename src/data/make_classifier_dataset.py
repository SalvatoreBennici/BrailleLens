import cv2
import functools
import yaml
import shutil
import random
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

AbsBBox = Tuple[int, int, int, int]
RANDOM_SEED: int = 42

def load_yolo_classes(yaml_path: Path) -> Dict[int, str]:
    if not yaml_path.exists(): return {}
    with open(yaml_path, 'r', encoding='utf-8') as f:
        names = yaml.safe_load(f).get('names', [])
    return {idx: str(name) for idx, name in (enumerate(names) if isinstance(names, list) else names.items())}

def yolo_to_abs_bbox(yolo_line: str, img_width: int, img_height: int, margin_pct: float = 0.0) -> Optional[AbsBBox]:
    parts = yolo_line.strip().split()
    if len(parts) < 5:
        return None
    _, x_center, y_center, width_norm, height_norm = map(float, parts[:5])

    box_width = round(width_norm * img_width * (1 + margin_pct))
    box_height = round(height_norm * img_height * (1 + margin_pct))
    if box_width == 0 or box_height == 0:
        return None

    x1 = round(x_center * img_width) - (box_width // 2)
    y1 = round(y_center * img_height) - (box_height // 2)
    return (x1, y1, x1 + box_width, y1 + box_height)

def compute_max_ioa(candidate: AbsBBox, references: List[AbsBBox]) -> float:
    """Compute the maximum Intersection over Area (IoA) of a candidate box against reference boxes."""
    cx1, cy1, cx2, cy2 = candidate
    candidate_area = (cx2 - cx1) * (cy2 - cy1)
    if candidate_area <= 0:
        return 0.0

    max_ioa = 0.0
    for rx1, ry1, rx2, ry2 in references:
        inter_width = max(0, min(cx2, rx2) - max(cx1, rx1))
        inter_height = max(0, min(cy2, ry2) - max(cy1, ry1))
        max_ioa = max(max_ioa, (inter_width * inter_height) / candidate_area)
    return max_ioa

def extract_and_save_crop(image: np.ndarray, bbox: AbsBBox, target_shape: Tuple[int, int], out_path: Path) -> bool:
    img_height, img_width = image.shape[:2]
    target_height, target_width = target_shape
    x1, y1, x2, y2 = bbox
    box_width, box_height = x2 - x1, y2 - y1

    # Adjust the shorter side to match the target aspect ratio
    if (box_width / box_height) > (target_width / target_height):
        box_height = round(box_width * (target_height / target_width))
    else:
        box_width = round(box_height * (target_width / target_height))

    center_x = x1 + (x2 - x1) // 2
    center_y = y1 + (y2 - y1) // 2
    new_x1 = center_x - (box_width // 2)
    new_y1 = center_y - (box_height // 2)
    new_x2 = new_x1 + box_width
    new_y2 = new_y1 + box_height

    crop = image[max(0, new_y1):min(img_height, new_y2), max(0, new_x1):min(img_width, new_x2)]
    if crop.size == 0:
        return False

    # Pad out-of-bounds regions using the crop's median color to blend seamlessly
    try:
        local_bg_color = int(np.median(crop))
    except Exception:
        local_bg_color = 128

    pad_top = max(0, -new_y1)
    pad_bottom = max(0, new_y2 - img_height)
    pad_left = max(0, -new_x1)
    pad_right = max(0, new_x2 - img_width)
    crop_padded = cv2.copyMakeBorder(
        crop, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=local_bg_color
    )

    interp = cv2.INTER_AREA if (box_width > target_width) else cv2.INTER_CUBIC
    cv2.imwrite(str(out_path), cv2.resize(crop_padded, (target_width, target_height), interpolation=interp))
    return True

def process_standard(
    image_path: Path,
    labels_dir: Path,
    output_dir: Path,
    target_shape: Tuple[int, int],
    class_map: Dict[int, str],
    margin_pct: float,
    prefix: str,
) -> int:
    label_path = labels_dir / image_path.with_suffix('.txt').name
    if not label_path.exists():
        return 0
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0
    img_height, img_width = image.shape[:2]
    count = 0
    for annotation_idx, line in enumerate(label_path.read_text().splitlines()):
        bbox = yolo_to_abs_bbox(line, img_width, img_height, margin_pct)
        if not bbox:
            continue
        class_id = int(line.split()[0])
        class_dir = output_dir / f"{(class_id + 1):02d}_{class_map.get(class_id, f'class_{class_id}')}"
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / f"{prefix}_{image_path.stem}_{annotation_idx}.jpg"
        if extract_and_save_crop(image, bbox, target_shape, out_path):
            count += 1
    return count

def process_dsbi_verso(
    image_path: Path,
    labels_dir: Path,
    output_dir: Path,
    target_shape: Tuple[int, int],
    margin_pct: float,
) -> int:
    verso_label_path = labels_dir / image_path.with_suffix('.txt').name
    recto_label_path = labels_dir / image_path.with_suffix('.txt').name.replace('+verso', '+recto')
    if not verso_label_path.exists():
        return 0
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0
    img_height, img_width = image.shape[:2]
    recto_lines = recto_label_path.read_text().splitlines() if recto_label_path.exists() else []
    recto_boxes = [b for b in (yolo_to_abs_bbox(line, img_width, img_height, 0.0) for line in recto_lines) if b]
    background_dir = output_dir / "00_background"
    background_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for annotation_idx, line in enumerate(verso_label_path.read_text().splitlines()):
        bbox = yolo_to_abs_bbox(line, img_width, img_height, margin_pct)
        if bbox and compute_max_ioa(bbox, recto_boxes) < 0.01:
            out_path = background_dir / f"DSBI_{image_path.stem}_{annotation_idx}.jpg"
            if extract_and_save_crop(image, bbox, target_shape, out_path):
                count += 1
    return count

@dataclass
class MiscCropParams:
    """Pre-computed parameters for a single random background crop, generated in the main thread."""
    image_path: Path
    crop_height_ratio: float
    x_offset_ratio: float
    y_offset_ratio: float


def process_misc(
    candidate_images: List[Path],
    output_dir: Path,
    target_shape: Tuple[int, int],
    num_crops: int,
) -> int:
    """Generate background crops from random regions of Angelina 'misc' images.

    All random parameters are pre-generated sequentially in the calling thread so that
    the output is fully deterministic regardless of thread scheduling.
    """
    background_dir = output_dir / "00_background"
    background_dir.mkdir(parents=True, exist_ok=True)

    # Pre-generate all random decisions in the main thread — determinism guaranteed.
    crop_params: List[MiscCropParams] = [
        MiscCropParams(
            image_path=random.choice(candidate_images),
            crop_height_ratio=random.uniform(0.05, 0.2),
            x_offset_ratio=random.random(),
            y_offset_ratio=random.random(),
        )
        for _ in range(num_crops)
    ]

    count = 0
    for crop_idx, params in enumerate(tqdm(crop_params, desc="Misc Backgrounds")):
        image = cv2.imread(str(params.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        img_height, img_width = image.shape[:2]
        crop_height = int(img_height * params.crop_height_ratio)
        crop_width = int(crop_height * (target_shape[1] / target_shape[0]))
        if crop_width >= img_width or crop_height >= img_height:
            continue
        x_offset = int(params.x_offset_ratio * (img_width - crop_width))
        y_offset = int(params.y_offset_ratio * (img_height - crop_height))
        bbox: AbsBBox = (x_offset, y_offset, x_offset + crop_width, y_offset + crop_height)
        out_path = background_dir / f"ANG_misc_{crop_idx}.jpg"
        if extract_and_save_crop(image, bbox, target_shape, out_path):
            count += 1
    return count

@dataclass
class DatasetProcessor:
    """Binds a dataset directory to its dispatch predicate and a pre-configured processing function.

    ``process_fn`` is a ``functools.partial`` with all static arguments pre-bound
    (target_shape, class_map, margin_pct, prefix). At call time it only receives
    ``(image_path, labels_dir, output_dir)``, keeping the dispatch loop uniform and branch-free.
    """
    dataset_dir: Path
    prefix: str
    predicate: Callable[[Path], bool]
    process_fn: Callable[..., int]


class ProcessorRegistry:
    """Maps image paths to the correct processing function (Strategy pattern).

    Adding support for a new dataset type means registering a new DatasetProcessor —
    no changes to the main processing loop are required (Open/Closed Principle).
    """

    def __init__(self) -> None:
        self._processors: List[DatasetProcessor] = []

    def register(self, processor: DatasetProcessor) -> None:
        self._processors.append(processor)

    def get_processor(self, image_path: Path, split: str) -> Optional[DatasetProcessor]:
        for processor in self._processors:
            if (image_path.is_relative_to(processor.dataset_dir / 'images' / split)
                    and processor.predicate(image_path)):
                return processor
        return None


def main() -> None:
    with open("configs/data_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    random.seed(cfg.get('seed', RANDOM_SEED))

    dir_out = Path(cfg['paths']['out'])
    merged_suffix: str = cfg['datasets']['merged']['suffix']
    cls_cfg = cfg['classifier']

    # Build dataset roots and prefix map from config — no hardcoded paths.
    # Processing order matters: Angelina must be first (its data.yaml is used for the class map).
    dataset_order = ['angelina', 'natural_scene', 'dsbi']
    dataset_roots: List[Path] = [
        dir_out / f"{cfg['datasets'][key]['name']}{merged_suffix}"
        for key in dataset_order
    ]
    # Strip trailing underscore: config stores "ANG_" for the detector; the classifier
    # adds its own "_" separator in the filename template, so it needs bare "ANG".
    prefix_map: Dict[str, str] = {
        f"{cfg['datasets'][key]['name']}{merged_suffix}": cfg['datasets'][key]['prefix'].rstrip('_')
        for key in dataset_order
    }

    output_root = Path(cls_cfg['crops_dir']).resolve()
    target_shape: Tuple[int, int] = tuple(cls_cfg['target_shape'])
    margin_pct: float = cls_cfg['margin_pct']
    misc_crops: int = cls_cfg['misc_crops']
    class_map = load_yolo_classes(dataset_roots[0] / "data.yaml")

    if output_root.exists():
        shutil.rmtree(output_root)

    # Build the registry once — dataset identity is split-independent.
    # To support a new dataset type, register a DatasetProcessor here; the loop below never changes.
    registry = ProcessorRegistry()
    for dataset_dir in dataset_roots:
        prefix = prefix_map.get(dataset_dir.name, "UNK")
        if prefix == "DSBI":
            registry.register(DatasetProcessor(
                dataset_dir=dataset_dir,
                prefix=prefix,
                predicate=lambda p: "+verso" in p.name,
                process_fn=functools.partial(
                    process_dsbi_verso,
                    target_shape=target_shape,
                    margin_pct=margin_pct,
                ),
            ))
        registry.register(DatasetProcessor(
            dataset_dir=dataset_dir,
            prefix=prefix,
            predicate=lambda _: True,
            process_fn=functools.partial(
                process_standard,
                target_shape=target_shape,
                class_map=class_map,
                margin_pct=margin_pct,
                prefix=prefix,
            ),
        ))

    for split in ["train", "val", "test"]:
        split_output_dir = output_root / split
        misc_images: List[Path] = []

        for dataset_dir in dataset_roots:
            prefix = prefix_map.get(dataset_dir.name, "UNK")
            images_dir = dataset_dir / "images" / split
            if not images_dir.exists():
                continue

            image_paths = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for image_path in image_paths:
                    if split == "train" and prefix == "ANG" and image_path.name.startswith("misc"):
                        misc_images.append(image_path)
                        continue

                    processor = registry.get_processor(image_path, split)
                    if processor is None:
                        continue

                    futures.append(executor.submit(
                        processor.process_fn,
                        image_path,
                        processor.dataset_dir / 'labels' / split,
                        split_output_dir,
                    ))

                for future in tqdm(as_completed(futures), total=len(futures), desc=f"[{split.upper()}] {dataset_dir.name}"):
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"\n[ERROR] {type(exc).__name__}: {exc}")

        if split == "train" and misc_images:
            process_misc(misc_images, split_output_dir, target_shape, misc_crops)


if __name__ == "__main__":
    main()