"""Train one model on FloodNet, RescueNet, or FWISD.

Example, from the repository root::

    python tools/train.py --model hdrnet --dataset floodnet --data-root D:/data/FloodNet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.protocol import PROTOCOL
from hdrnet.builder import MODELS, build_criterion, build_model
from hdrnet.data import make_loader
from hdrnet.engine import learning_rate, set_learning_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a comparison or HDRNet model")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--dataset", required=True, choices=("floodnet", "rescuenet", "fwisd"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--iters", type=int, default=PROTOCOL["max_iters"])
    parser.add_argument("--batch-size", type=int, default=PROTOCOL["batch_size_per_gpu"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=PROTOCOL["lr"])
    parser.add_argument("--crop-size", type=int, default=PROTOCOL["crop_size"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--ckpt-interval", type=int, default=4000)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    work_dir = args.work_dir or Path("work_dirs") / f"{args.model}_{args.dataset}"
    work_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model, args.dataset).to(device)
    criterion = build_criterion(args.model, args.dataset).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=PROTOCOL["betas"], weight_decay=PROTOCOL["weight_decay"])
    loader = make_loader(
        args.dataset,
        args.data_root,
        train=True,
        batch_size=args.batch_size,
        workers=args.workers,
        class_aware=args.model == "da-segformer",
        crop_size=args.crop_size,
    )

    start = 0
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = int(state.get("step", 0))

    model.train()
    data_iter = iter(loader)
    for step in range(start, args.iters):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        lr = learning_rate(step, args.lr, max_iters=args.iters, power=PROTOCOL["power"])
        set_learning_rate(optimizer, lr)
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(image)
        loss = criterion(output, mask)
        value = loss["loss"] if isinstance(loss, dict) else loss
        value.backward()
        optimizer.step()
        if step % args.log_interval == 0 or step + 1 == args.iters:
            print(f"iter {step:06d}/{args.iters}  lr {lr:.6e}  loss {value.item():.4f}", flush=True)
        if (step + 1) % args.ckpt_interval == 0 or step + 1 == args.iters:
            path = work_dir / f"iter_{step + 1}.pth"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step + 1, "args": vars(args)}, path)
            print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
