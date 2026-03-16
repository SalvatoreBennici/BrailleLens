import cv2
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(mean: list[float], std: list[float], is_train: bool) -> A.Compose:
    if is_train:
        return A.Compose([
            A.Affine(
                scale=(0.95, 1.05), 
                translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)}, 
                rotate=(-5, 5), 
                p=0.5
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, 
                contrast_limit=(-0.3, 0.3), 
                p=0.5
            ),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.4),
            A.MotionBlur(blur_limit=3, p=0.2), 
            A.CoarseDropout(
                num_holes_range=(1, 2), 
                hole_height_range=(4, 6), 
                hole_width_range=(4, 6), 
                fill=128, 
                p=0.3
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
        
    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

class BrailleMultiLabelDataset(Dataset):
    def __init__(self, data_dir: Path, transforms: A.Compose, is_train: bool = False, cache_in_ram: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        self.is_train = is_train
        self.cache_in_ram = cache_in_ram
        
        self.image_paths: list[Path] = []
        self.target_tensors: list[torch.Tensor] = []
        self.ram_cache: list[np.ndarray] = []
        
        self._build_dataset()

    def _parse_braille_string(self, class_name: str) -> torch.Tensor:
        target_tensor = torch.zeros(6, dtype=torch.float32)
        for char in class_name:
            if char.isdigit() and 1 <= int(char) <= 6:
                target_tensor[int(char) - 1] = 1.0
        return target_tensor

    def _build_dataset(self) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
            
        for class_dir in [d for d in self.data_dir.iterdir() if d.is_dir()]:
            folder_parts = class_dir.name.split('_')
            if len(folder_parts) < 2: 
                continue

            target_tensor = self._parse_braille_string(folder_parts[1].strip())
            images = list(class_dir.glob("*.jpg"))
            
            self.image_paths.extend(images)
            self.target_tensors.extend([target_tensor] * len(images))

            if self.cache_in_ram:
                for img_path in images:
                    image_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if image_gray is None:
                        raise ValueError(f"Unreadable image: {img_path}")
                    self.ram_cache.append(np.expand_dims(image_gray, axis=-1))

    def _apply_geometric_flips(self, image: np.ndarray, target: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)
            target = target[[3, 4, 5, 0, 1, 2]]

        return image, target

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        target_tensor = self.target_tensors[index]

        if self.cache_in_ram:
            image_gray = self.ram_cache[index]
        else:
            image_gray = cv2.imread(str(self.image_paths[index]), cv2.IMREAD_GRAYSCALE)
            image_gray = np.expand_dims(image_gray, axis=-1)

        if self.is_train:
            image_gray, target_tensor = self._apply_geometric_flips(image_gray, target_tensor)

        augmented = self.transforms(image=image_gray)
        return augmented['image'], target_tensor