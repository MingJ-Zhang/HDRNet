"""Predict one RGB image and export an indexed mask plus color overlay."""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.datasets import DATASETS
from configs.protocol import PROTOCOL
from hdrnet.builder import MODELS, build_model
from hdrnet.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from hdrnet.data.transforms import Normalize
from hdrnet.engine import slide_inference
from hdrnet.runtime import load_checkpoint, resolve_device, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    p.add_argument("--model", choices=sorted(MODELS), default="hdrnet")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/prediction"))
    p.add_argument("--crop-size", type=int, default=PROTOCOL["slide_crop"])
    p.add_argument("--stride", type=int, default=PROTOCOL["slide_stride"])
    a = p.parse_args()
    device = resolve_device(a.device)
    model = build_model(a.model, a.dataset).to(device).eval()
    load_checkpoint(a.checkpoint, model, a.model, a.dataset)
    with Image.open(a.image) as image:
        rgb = np.array(image.convert("RGB"))
    normalized = Normalize(IMAGENET_MEAN, IMAGENET_STD)(rgb).unsqueeze(0).to(device)
    logits = slide_inference(model, normalized, len(DATASETS[a.dataset]["classes"]), a.crop_size, a.stride)
    mask = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    palette = np.asarray(DATASETS[a.dataset]["palette"], dtype=np.uint8)
    colored = palette[mask]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(a.output_dir / "mask.png")
    Image.fromarray(colored).save(a.output_dir / "color.png")
    Image.fromarray(((rgb.astype(np.float32) + colored) / 2).astype(np.uint8)).save(a.output_dir / "overlay.png")
    write_json(a.output_dir / "prediction.json", {"model": a.model, "dataset": a.dataset,
        "image": str(a.image), "checkpoint_sha256": sha256(a.checkpoint), "classes": DATASETS[a.dataset]["classes"],
        "shape": list(mask.shape), "inference": {"crop": a.crop_size, "stride": a.stride}})
    print(f"saved indexed mask, color map and overlay to {a.output_dir}")


if __name__ == "__main__":
    main()
