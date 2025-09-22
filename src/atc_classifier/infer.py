import argparse
import torch
from PIL import Image
import torchvision.transforms as T
from .models.model import ATCModel
from .config import load_config


def predict(cfg_path: str, checkpoint: str, image_path: str):
    cfg = load_config(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ATCModel(
        backbone_name=cfg.model.backbone,
        pretrained=False,
        num_classes=cfg.data.num_classes if cfg.model.classification_head else 0,
        regression_traits=cfg.model.regression_traits,
    ).to(device)

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    tfm = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)

    result = {}
    if "class_logits" in out:
        pred = out["class_logits"].softmax(dim=-1).argmax(dim=-1).item()
        result["species"] = "buffalo" if pred == 1 else "cattle"
    if "traits" in out:
        result["traits"] = out["traits"].cpu().squeeze(0).tolist()

    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    args = p.parse_args()
    predict(args.config, args.checkpoint, args.image)
