import argparse
import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .config import load_config
from .data.datasets import ATCDataset
from .models.model import ATCModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(cfg) -> Dict[str, DataLoader]:
    common_tfms = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ATCDataset(cfg.data.train_csv, cfg.data.image_root, transform=common_tfms, traits=cfg.data.traits)
    val_ds = ATCDataset(cfg.data.val_csv, cfg.data.image_root, transform=common_tfms, traits=cfg.data.traits)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    return {"train": train_loader, "val": val_loader}


def train(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    set_seed(cfg.seed)
    os.makedirs(cfg.train.output_dir, exist_ok=True)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    loaders = build_loaders(cfg)

    model = ATCModel(
        backbone_name=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        num_classes=cfg.data.num_classes if cfg.model.classification_head else 0,
        regression_traits=cfg.model.regression_traits,
    ).to(device)

    ce_loss = nn.CrossEntropyLoss() if cfg.model.classification_head else None
    mse_loss = nn.MSELoss() if cfg.model.regression_traits else None

    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    best_val = float("inf")

    for epoch in range(cfg.train.num_epochs):
        model.train()
        total_loss = 0.0
        for images, targets in loaders["train"]:
            images = images.to(device)
            class_t = targets["class"].to(device)
            traits_t = targets["traits"]
            traits_t = traits_t.to(device) if traits_t is not None else None

            optimizer.zero_grad()
            out = model(images)

            loss = 0.0
            if ce_loss is not None and "class_logits" in out:
                loss = loss + ce_loss(out["class_logits"], class_t)
            if mse_loss is not None and (traits_t is not None) and ("traits" in out):
                loss = loss + mse_loss(out["traits"], traits_t)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # simple val pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in loaders["val"]:
                images = images.to(device)
                class_t = targets["class"].to(device)
                traits_t = targets["traits"]
                traits_t = traits_t.to(device) if traits_t is not None else None

                out = model(images)
                loss = 0.0
                if ce_loss is not None and "class_logits" in out:
                    loss = loss + ce_loss(out["class_logits"], class_t)
                if mse_loss is not None and (traits_t is not None) and ("traits" in out):
                    loss = loss + mse_loss(out["traits"], traits_t)
                val_loss += loss.item()

        avg_train = total_loss / max(len(loaders["train"]), 1)
        avg_val = val_loss / max(len(loaders["val"]), 1)
        print(f"Epoch {epoch+1}/{cfg.train.num_epochs} - train_loss={avg_train:.4f} val_loss={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), os.path.join("models", "best.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
