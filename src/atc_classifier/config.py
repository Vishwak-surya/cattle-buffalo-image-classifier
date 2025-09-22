from typing import List
from pydantic import BaseModel
import yaml


class DataConfig(BaseModel):
    train_csv: str
    val_csv: str
    image_root: str
    num_classes: int = 2
    traits: List[str] = []


class ModelConfig(BaseModel):
    backbone: str = "resnet50"
    pretrained: bool = True
    classification_head: bool = True
    regression_traits: List[str] = []


class TrainConfig(BaseModel):
    batch_size: int = 16
    num_epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = "cuda"
    output_dir: str = "outputs"


class InferenceConfig(BaseModel):
    checkpoint_path: str = "models/best.pt"


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class Config(BaseModel):
    project_name: str = "cattle-buffalo-image-classifier"
    seed: int = 42
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    inference: InferenceConfig
    api: APIConfig


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
