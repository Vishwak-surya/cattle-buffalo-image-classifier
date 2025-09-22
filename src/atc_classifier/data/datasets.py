from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch
import os


class ATCDataset(Dataset):
    """
    Expects CSV with columns:
    - image: relative path to image file under image_root
    - label: 0 (cattle) or 1 (buffalo) [optional]
    - optional columns for traits (e.g., body_length, height_withers, ...)
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        transform=None,
        traits: Optional[List[str]] = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform
        self.traits = traits or []

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Optional[torch.Tensor]]]:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, str(row["image"]))
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["label"]) if "label" in row else 0
        y_class = torch.tensor(label, dtype=torch.long)

        y_traits = None
        if self.traits:
            traits_vals = [float(row.get(t, 0.0)) for t in self.traits]
            y_traits = torch.tensor(traits_vals, dtype=torch.float32)

        return image, {"class": y_class, "traits": y_traits}
