import cv2
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

def _angelina_stem(filename: str) -> str:
    """Strip Angelina-specific suffixes (.labeled.jpg, .jpg) to get the base stem."""
    return filename.replace('.labeled.jpg', '').replace('.jpg', '')


def _dots_from_bitmask(raw_id: int) -> str:
    """Convert a 6-bit integer to a Braille dot string (e.g. 0b000101 → '13')."""
    return "".join(str(i + 1) for i in range(6) if raw_id & (1 << i))


def _dots_from_flags(flags: List[int]) -> str:
    """Convert a list of 6 binary flags to a Braille dot string (e.g. [1,0,1,0,0,0] → '13')."""
    return "".join(str(i + 1) for i, d in enumerate(flags) if d == 1)


@dataclass
class YoloBBox:
    class_id: int
    xc: float
    yc: float
    w: float
    h: float

    def to_txt_line(self) -> str:
        return f"{self.class_id} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"

def to_yolo_format(
    xmin: float, ymin: float, xmax: float, ymax: float,
    img_width: int, img_height: int,
) -> Tuple[float, float, float, float]:
    norm_w = (xmax - xmin) / img_width
    norm_h = (ymax - ymin) / img_height
    xc = (xmin + xmax) / 2.0 / img_width
    yc = (ymin + ymax) / 2.0 / img_height
    return (
        max(0.0, min(xc, 1.0)),
        max(0.0, min(yc, 1.0)),
        max(0.0, min(norm_w, 1.0)),
        max(0.0, min(norm_h, 1.0)),
    )

class DatasetConverter:
    def __init__(self, raw_path: Path, out_path: Path, splits: List[str], unified_classes: List[str], val_split_ratio: float = 0.20):
        self.raw_path = raw_path
        self.out_path = out_path
        self.splits = splits
        self.unified_classes = unified_classes
        self.num_classes = len(unified_classes)
        self.val_split_ratio = val_split_ratio
        self.total_chars = 0
        self._class_lookup: Dict[str, int] = {name: idx for idx, name in enumerate(unified_classes)}

        self._setup_directories()
        self._write_data_yaml()

    def _setup_directories(self) -> None:
        if self.out_path.exists():
            shutil.rmtree(self.out_path)
        for split in self.splits:
            (self.out_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.out_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    def _write_data_yaml(self) -> None:
        yaml_content = f"""train: images/train
val: images/val
test: images/test

nc: {self.num_classes}
names: {self.unified_classes}
"""
        (self.out_path / 'data.yaml').write_text(yaml_content, encoding='utf-8')

    def _resolve_class_id(self, dots_str: str) -> Optional[int]:
        """Return the class index for a dot string, or None if not in the class list."""
        return self._class_lookup.get(dots_str)

    def _write_yolo_labels(self, split: str, img_src: Path, bboxes: List[YoloBBox]) -> None:
        safe_stem = f"{img_src.parent.name}_{img_src.stem}"
        img_dst = self.out_path / 'images' / split / f"{safe_stem}{img_src.suffix}"
        lbl_dst = self.out_path / 'labels' / split / f"{safe_stem}.txt"
        
        shutil.copy2(img_src, img_dst)
        self.total_chars += len(bboxes)
        
        with open(lbl_dst, 'w', encoding='utf-8') as f:
            for bbox in bboxes:
                f.write(bbox.to_txt_line() + '\n')

    def process(self) -> int:
        raise NotImplementedError


class AngelinaConverter(DatasetConverter):
    DIR_IGNORE = 'not_braille'
    DELIMITER = ';'

    def process(self) -> int:
        if not self.raw_path.exists():
            print(f"Error: Path {self.raw_path} not found.")
            return 0
            
        img_dict = {}
        for img_path in self.raw_path.rglob("*.jpg"):
            stem = _angelina_stem(img_path.name)
            img_dict[stem] = img_path

        splits_map: Dict[str, List[Path]] = {split: [] for split in self.splits}
        for split_name in splits_map.keys():
            for txt_file in self.raw_path.rglob(f"{split_name}.txt"):
                lines = txt_file.read_text('utf-8').splitlines()
                for line in lines:
                    if not line.strip(): 
                        continue
                    base_stem = _angelina_stem(line.replace('\\', '/').split('/')[-1])
                    if base_stem in img_dict:
                        splits_map[split_name].append(img_dict[base_stem])

        for split, paths in splits_map.items():
            for img_path in paths:
                bboxes = self._parse_csv(img_path)
                self._write_yolo_labels(split, img_path, bboxes)
        
        return self.total_chars

    def _parse_csv(self, img_path: Path) -> List[YoloBBox]:
        if self.DIR_IGNORE in img_path.parts:
            return []
        
        base_stem = _angelina_stem(img_path.name)
        csv_path = img_path.parent / f"{base_stem}.labeled.csv"
        
        if not csv_path.exists():
            csv_path = img_path.parent / f"{base_stem}.csv"
            if not csv_path.exists():
                return []

        bboxes = []
        for line in csv_path.read_text('utf-8').splitlines():
            if not line.strip(): 
                continue
            parts = line.split(self.DELIMITER)
            if len(parts) < 5: 
                continue
            
            left, top, right, bottom = map(float, parts[:4])
            raw_id = int(parts[4])
            if raw_id < 1 or raw_id > 63: 
                continue
            
            dots_str = _dots_from_bitmask(raw_id)
            class_id = self._resolve_class_id(dots_str)
            if class_id is None:
                continue
            
            w = right - left
            h = bottom - top
            xc = left + w / 2.0
            yc = top + h / 2.0
            
            bboxes.append(YoloBBox(class_id, xc, yc, w, h))
            
        return bboxes


class NaturalSceneConverter(DatasetConverter):
    DIR_VOC = 'voc-data'
    DIR_MAIN = 'ImageSets/Main'
    DIR_IMAGES_VOC = 'JPEGImages'
    DIR_ANNOTATIONS = 'Annotations'

    def process(self) -> int:
        if not self.raw_path.exists():
            print(f"Error: Path {self.raw_path} not found.")
            return 0
            
        voc_dir = self.raw_path / self.DIR_VOC
        main_dir = voc_dir / self.DIR_MAIN
        
        train_txt = main_dir / "train.txt"
        test_txt = main_dir / "test.txt"
        
        if not train_txt.exists() or not test_txt.exists():
            print(f"Error: Txt files not found in {main_dir}")
            return 0
        
        train_names = [line.strip() for line in train_txt.read_text('utf-8').splitlines() if line.strip()]
        test_names = [line.strip() for line in test_txt.read_text('utf-8').splitlines() if line.strip()]

        random.shuffle(train_names)
        val_split_idx = int(len(train_names) * self.val_split_ratio)
        val_names = train_names[:val_split_idx]
        train_names = train_names[val_split_idx:]

        self._process_split("train", train_names, voc_dir)
        self._process_split("val", val_names, voc_dir)
        self._process_split("test", test_names, voc_dir)

        return self.total_chars

    def _process_split(self, split: str, names: List[str], voc_dir: Path) -> None:
        for name in names:
            img_path = voc_dir / self.DIR_IMAGES_VOC / f"{name}.jpg"
            xml_path = voc_dir / self.DIR_ANNOTATIONS / f"{name}.xml"
            
            if not img_path.exists() or not xml_path.exists():
                continue
                
            bboxes = self._parse_xml(xml_path)
            self._write_yolo_labels(split, img_path, bboxes)

    def _parse_xml(self, xml_path: Path) -> List[YoloBBox]:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        if size is None:
            return []
        width_el, height_el = size.find('width'), size.find('height')
        if width_el is None or height_el is None:
            return []
        img_width, img_height = int(width_el.text), int(height_el.text)

        bboxes = []
        for obj in root.findall('object'):
            name_el = obj.find('name')
            if name_el is None:
                continue
            class_id = int(name_el.text) - 1
            if class_id < 0 or class_id >= self.num_classes:
                continue

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            xc, yc, norm_w, norm_h = to_yolo_format(xmin, ymin, xmax, ymax, img_width, img_height)
            bboxes.append(YoloBBox(class_id, xc, yc, norm_w, norm_h))
            
        return bboxes


class DSBIConverter(DatasetConverter):
    IMAGE_VARIANTS: Tuple[str, ...] = ('+recto', '+verso')
    
    def process(self) -> int:
        if not self.raw_path.exists():
            print(f"Error: Path {self.raw_path} not found.")
            return 0
            
        train_txt = self.raw_path / "train.txt"
        test_txt = self.raw_path / "test.txt"
        
        if not train_txt.exists() or not test_txt.exists():
            print(f"Error: Txt files not found in {self.raw_path}")
            return 0
            
        train_lines = [line.strip().replace('\\', '/') for line in train_txt.read_text('utf-8').splitlines() if line.strip()]
        test_lines = [line.strip().replace('\\', '/') for line in test_txt.read_text('utf-8').splitlines() if line.strip()]
        
        random.shuffle(train_lines)
        val_split_idx = int(len(train_lines) * self.val_split_ratio)
        val_lines = train_lines[:val_split_idx]
        train_lines = train_lines[val_split_idx:]
        
        variant_dict: Dict[str, Path] = {}
        for img_path in self.raw_path.rglob("*.jpg"):
            variant_dict[img_path.name] = img_path

        self._process_split("train", train_lines, variant_dict)
        self._process_split("val", val_lines, variant_dict)
        self._process_split("test", test_lines, variant_dict)
        
        return self.total_chars

    def _process_split(self, split_name: str, file_lines: List[str], variant_dict: Dict[str, Path]) -> None:
        for line in file_lines:
            base_stem = line.replace('\\', '/').split('/')[-1].replace('.jpg', '')
            
            for suffix in self.IMAGE_VARIANTS:
                variant_filename = f"{base_stem}{suffix}.jpg"
                
                if variant_filename in variant_dict:
                    variant_img_path = variant_dict[variant_filename]
                    variant_txt_path = variant_img_path.with_suffix('.txt')
                    
                    if variant_txt_path.exists():
                        bboxes = self._parse_dsbi_txt(variant_txt_path, variant_img_path)
                        self._write_yolo_labels(split_name, variant_img_path, bboxes)

    def _parse_dsbi_txt(self, txt_path: Path, img_path: Path) -> List[YoloBBox]:
        if txt_path.stat().st_size == 0:
            return []
            
        lines = txt_path.read_text('utf-8').splitlines()
        if len(lines) < 4:
            return []
            
        img = cv2.imread(str(img_path))
        if img is None:
            return []
        img_height, img_width = img.shape[:2]

        try:
            vertical_lines = [int(x) for x in lines[1].split()]
            horizontal_lines = [int(x) for x in lines[2].split()]
        except ValueError:
            return []

        bboxes = []
        for line in lines[3:]:
            parts = [int(x) for x in line.split()]
            if len(parts) < 8: 
                continue
            
            row, col = parts[0], parts[1]
            dots_flags = parts[2:8]
            dots_str = _dots_from_flags(dots_flags)
            if not dots_str:
                continue
            class_id = self._resolve_class_id(dots_str)
            if class_id is None:
                continue

            try:
                xmin = vertical_lines[(col - 1) * 2]
                xmax = vertical_lines[(col - 1) * 2 + 1]
                ymin = horizontal_lines[(row - 1) * 3]
                ymax = horizontal_lines[(row - 1) * 3 + 2]
            except IndexError:
                try:
                    xmin = vertical_lines[col - 1]
                    xmax = vertical_lines[col]
                    ymin = horizontal_lines[(row - 1) * 3]
                    ymax = horizontal_lines[row * 3]
                except IndexError:
                    continue

            norm_xc, norm_yc, norm_w, norm_h = to_yolo_format(xmin, ymin, xmax, ymax, img_width, img_height)
            bboxes.append(YoloBBox(class_id, norm_xc, norm_yc, norm_w, norm_h))
            
        return bboxes