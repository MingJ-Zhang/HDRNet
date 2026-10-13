"""Sliding-window evaluation.

Example::

    python tools/test.py --model hdrnet --dataset floodnet ^
        --data-root D:/data/FloodNet --checkpoint work_dirs/hdrnet_floodnet/iter_80000.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.datasets import DATASETS
from configs.protocol import PROTOCOL
from hdrnet.builder import MODELS, build_model
from hdrnet.data import make_loader
from hdrnet.engine import ConfusionMeter, slide_inference
from hdrnet.runtime import environment, load_checkpoint, resolve_device, sha256, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate with a sliding window")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--crop-size", type=int, default=PROTOCOL["slide_crop"])
    parser.add_argument("--stride", type=int, default=PROTOCOL["slide_stride"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    parser.add_argument("--save-predictions", type=Path, help="directory for indexed PNG masks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    spec = DATASETS[args.dataset]
    num_classes = len(spec["classes"])
    model = build_model(args.model, args.dataset).to(device)
    load_checkpoint(args.checkpoint, model, args.model, args.dataset)
    model.eval()
    loader = make_loader(args.dataset, args.data_root, train=False, batch_size=1, workers=args.workers)
    meter = ConfusionMeter(num_classes)
    for index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        logits = slide_inference(model, image, num_classes, args.crop_size, args.stride)
        pred = logits.argmax(dim=1).cpu()
        meter.update(pred, batch["mask"])
        if args.save_predictions:
            args.save_predictions.mkdir(parents=True, exist_ok=True)
            # Index prefix avoids collisions when different subdirectories share a stem.
            path = args.save_predictions / f"{index:06d}_{batch['name'][0]}.png"
            Image.fromarray(pred[0].numpy().astype(np.uint8)).save(path)
        if index % 20 == 0:
            print(f"[{index + 1}/{len(loader)}] {batch['name'][0]}", flush=True)
    scores = meter.compute()
    for row, name in zip(scores["per_class"], spec["classes"]):
        row["name"] = name
    write_json(args.output, {"model": args.model, "dataset": args.dataset,
        "checkpoint_sha256": sha256(args.checkpoint), "environment": environment(),
        "inference": {"crop": args.crop_size, "stride": args.stride}, "scores": scores})
    print(
        f"{args.model} {args.dataset}  mIoU {scores['mIoU']:.2f}  "
        f"mF1 {scores['mF1']:.2f}  paper-mF1 {scores['paper_mF1']:.2f}"
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
