import argparse
import csv
import os
from pathlib import Path


def scan_images_to_csv(image_root: str, out_csv: str) -> None:
    image_root = Path(image_root)
    rows = []
    for p in image_root.rglob("*.jpg"):
        rel = p.relative_to(image_root).as_posix()
        rows.append({"image": rel, "label": ""})
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label"]) 
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="Path to image root directory")
    p.add_argument("--out", required=True, help="Path to output CSV")
    args = p.parse_args()
    scan_images_to_csv(args.images, args.out)
