import yaml
import random
import shutil
import itertools
from pathlib import Path
from typing import List, Tuple

from converters import AngelinaConverter, NaturalSceneConverter, DSBIConverter

def generate_unified_classes() -> List[str]:
    classes = []
    for r in range(1, 7):
        for combo in itertools.combinations("123456", r):
            classes.append("".join(combo))
    return classes

def _unique_stem(base_name: str, directory: Path, suffix: str) -> str:
    """Return a unique filename stem, appending _copy{n} if the base already exists."""
    if not (directory / f"{base_name}{suffix}").exists():
        return base_name
    for i in itertools.count(1):
        candidate = f"{base_name}_copy{i}"
        if not (directory / f"{candidate}{suffix}").exists():
            return candidate


def create_merged_dataset(
    datasets_to_merge: List[Tuple[Path, str, Tuple[str, ...]]], 
    merged_dir: Path, 
    splits: List[str], 
    unified_classes: List[str]
) -> None:
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
        
    for split in splits:
        (merged_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (merged_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    yaml_content = f"train: images/train\nval: images/val\ntest: images/test\n\nnc: {len(unified_classes)}\nnames: {unified_classes}\n"
    (merged_dir / 'data.yaml').write_text(yaml_content, encoding='utf-8')

    for ds_dir, prefix, exclude_substrings in datasets_to_merge:
        for split in splits:
            for img_path in (ds_dir / 'images' / split).glob("*.jpg"):
                
                if any(substring in img_path.name for substring in exclude_substrings):
                    continue
                
                lbl_path = ds_dir / 'labels' / split / f"{img_path.stem}.txt"
                
                dst_base_name = f"{prefix}{img_path.stem}"
                dst_safe_name = _unique_stem(dst_base_name, merged_dir / 'images' / split, img_path.suffix)
                
                dst_img = merged_dir / 'images' / split / f"{dst_safe_name}{img_path.suffix}"
                dst_lbl = merged_dir / 'labels' / split / f"{dst_safe_name}.txt"
                
                shutil.copy2(img_path, dst_img)
                if lbl_path.exists():
                    shutil.copy2(lbl_path, dst_lbl)

def balance_natural_scene_in_merged(
    merged_dir: Path, oversample_ratio: float, prefix_ns: str, prefixes_others: Tuple[str, ...]
) -> None:
    split = 'train'
    img_dir = merged_dir / 'images' / split
    lbl_dir = merged_dir / 'labels' / split
    
    if not img_dir.exists() or not lbl_dir.exists():
        return
        
    # Cache (image_path, char_count) during the scan pass — char counts are available at zero extra I/O cost
    # in the oversampling loop below, eliminating repeated label-file reads.
    ang_dsbi_chars = 0
    ns_chars = 0
    ns_images: List[Tuple[Path, int]] = []

    for img_path in img_dir.glob("*.jpg"):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        chars_in_img = len(lbl_path.read_text('utf-8').splitlines())

        if img_path.name.startswith(prefix_ns):
            ns_chars += chars_in_img
            ns_images.append((img_path, chars_in_img))
        elif img_path.name.startswith(prefixes_others):
            ang_dsbi_chars += chars_in_img

    if not ns_images or ang_dsbi_chars == 0:
        return

    target_ratio = oversample_ratio / (1.0 - oversample_ratio)
    target_ns_chars = int(ang_dsbi_chars * target_ratio)
    chars_needed = target_ns_chars - ns_chars

    aug_idx = 1
    while chars_needed > 0:
        src_img, chars_in_img = random.choice(ns_images)
        src_lbl = lbl_dir / f"{src_img.stem}.txt"

        if not src_lbl.exists() or chars_in_img == 0:
            continue

        safe_name = f"{src_img.stem}_aug_{aug_idx}"
        while (img_dir / f"{safe_name}.jpg").exists():
            aug_idx += 1
            safe_name = f"{src_img.stem}_aug_{aug_idx}"

        shutil.copy2(src_img, img_dir / f"{safe_name}.jpg")
        shutil.copy2(src_lbl, lbl_dir / f"{safe_name}.txt")

        chars_needed -= chars_in_img
        aug_idx += 1

def main() -> None:
    config_path = Path("configs/data_config.yaml")
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    random.seed(config['seed'])

    dir_raw = Path(config['paths']['raw'])
    dir_out = Path(config['paths']['out'])
    splits = config['detector']['splits']
    val_split_ratio = config['detector']['val_split_ratio']
    unified_classes = generate_unified_classes()

    ang_cfg = config['datasets']['angelina']
    ns_cfg = config['datasets']['natural_scene']
    dsbi_cfg = config['datasets']['dsbi']
    merged_cfg = config['datasets']['merged']
    
    angelina_raw = dir_raw / ang_cfg['name']
    angelina_out = dir_out / f"{ang_cfg['name']}{merged_cfg['suffix']}"

    ns_raw = dir_raw / ns_cfg['name']
    ns_out = dir_out / f"{ns_cfg['name']}{merged_cfg['suffix']}"

    dsbi_raw = dir_raw / dsbi_cfg['name']
    dsbi_out = dir_out / f"{dsbi_cfg['name']}{merged_cfg['suffix']}"

    merged_out = dir_out / f"{merged_cfg['name']}{merged_cfg['suffix']}"

    print(f"Processing {ang_cfg['name']}...")
    AngelinaConverter(angelina_raw, angelina_out, splits, unified_classes, val_split_ratio).process()
    
    print(f"Processing {ns_cfg['name']}...")
    NaturalSceneConverter(ns_raw, ns_out, splits, unified_classes, val_split_ratio).process()

    print(f"Processing {dsbi_cfg['name']}...")
    DSBIConverter(dsbi_raw, dsbi_out, splits, unified_classes, val_split_ratio).process()

    print("Creating Merged Dataset...")
    datasets_to_merge = [
        (angelina_out, ang_cfg['prefix'], ()),           
        (ns_out, ns_cfg['prefix'], ()),                  
        (dsbi_out, dsbi_cfg['prefix'], ('+verso',)) # No verso for the detector
    ]
    create_merged_dataset(datasets_to_merge, merged_out, splits, unified_classes)
    
    print("Applying Training Oversampling...")
    balance_natural_scene_in_merged(
        merged_dir=merged_out,
        oversample_ratio=config['detector']['ns_oversample_ratio'],
        prefix_ns=ns_cfg['prefix'],
        prefixes_others=(ang_cfg['prefix'], dsbi_cfg['prefix'])
    )
    print("Dataset pipeline completed successfully.")

if __name__ == '__main__':
    main()