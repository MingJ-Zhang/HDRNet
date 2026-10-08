"""Sliding-window evaluation.

Example::

    python tools/test.py --model hdrnet --dataset floodnet ^
        --data-root D:/data/FloodNet --checkpoint work_dirs/hdrnet_floodnet/iter_80000.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.datasets import DATASETS
from configs.protocol import PROTOCOL
from hdrnet.builder import MODELS, build_model
from hdrnet.data import make_loader
from hdrnet.engine import ConfusionMeter, slide_inference


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    spec = DATASETS[args.dataset]
    num_classes = len(spec["classes"])
    model = build_model(args.model, args.dataset).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()
    loader = make_loader(args.dataset, args.data_root, train=False, batch_size=1, workers=args.workers)
    meter = ConfusionMeter(num_classes)
    for index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        logits = slide_inference(model, image, num_classes, args.crop_size, args.stride)
        pred = logits.argmax(dim=1).cpu()
        meter.update(pred, batch["mask"])
        if index % 20 == 0:
            print(f"[{index + 1}/{len(loader)}] {batch['name'][0]}", flush=True)
    scores = meter.compute()
    print(
        f"{args.model} {args.dataset}  mIoU {scores['mIoU']:.2f}  "
        f"mF1 {scores['mF1']:.2f}  paper-mF1 {scores['paper_mF1']:.2f}"
    )


if __name__ == "__main__":
    main()
