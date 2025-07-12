"""
Data loaders for bird detection datasets.
"""

import os
import yaml
from typing import Dict, Any, List, Optional, Callable, Tuple
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms


def get_raw_data_config(set_id: int) -> Dict[str, Any]:
    """
    Load configuration for a specific dataset set.
    
    Args:
        set_id: The set ID to load configuration for
        
    Returns:
        Dictionary containing the set configuration
    """
    config_path = Path(f"data/raw/set_{set_id}.yaml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


class SimpleBirdDataset(Dataset):
    """
    Dataset for bird detection from a directory of JPG images.
    Returns (image_tensor, filename).
    """
    def __init__(self, data_dir: str, image_size: int, transform: Optional[Callable] = None):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.filenames = sorted([
            f for f in os.listdir(self.data_dir)
            if f.lower().endswith('.jpg')
        ])
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                # Optionally add normalization here if your model expects it
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        img_name = self.filenames[idx]
        img_path = self.data_dir / img_name
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, img_name

class MultiSetBirdDataset(Dataset):
    """
    Dataset for bird detection from multiple directories of JPG images.
    Returns (image_tensor, filename).
    """
    def __init__(self, data_dirs: list, image_size: int, transform: Optional[Callable] = None):
        self.data_dirs = [Path(d) for d in data_dirs]
        self.image_size = image_size
        self.filenames = []
        self.filepaths = []
        for d in self.data_dirs:
            for f in sorted(os.listdir(d)):
                if f.lower().endswith('.jpg'):
                    self.filenames.append(f)
                    self.filepaths.append(d / f)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                # Optionally add normalization here if your model expects it
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        img_path = self.filepaths[idx]
        img_name = img_path.name
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, img_name 