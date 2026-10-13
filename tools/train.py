"""Iteration-based training, compatible with python and torchrun."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.datasets import DATASETS
from configs.protocol import PROTOCOL
from hdrnet.builder import MODELS, build_criterion, build_model
from hdrnet.data import make_dataset
from hdrnet.data.prior import estimate_prior
from hdrnet.engine import learning_rate, set_learning_rate
from hdrnet.runtime import (capture_rng, environment, load_checkpoint, resolve_device, restore_rng,
                            save_checkpoint, seed_everything, seed_worker, write_json)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--work-dir", type=Path)
    p.add_argument("--iters", type=int, default=PROTOCOL["max_iters"])
    p.add_argument("--batch-size", type=int, default=PROTOCOL["batch_size_per_gpu"], help="per process")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=PROTOCOL["lr"])
    p.add_argument("--crop-size", type=int, default=PROTOCOL["crop_size"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--amp", action="store_true", help="optional CUDA FP16; default is FP32")
    p.add_argument("--sync-bn", action="store_true")
    p.add_argument("--grad-clip", type=float, default=0.)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--ckpt-interval", type=int, default=4000)
    p.add_argument("--resume", type=Path)
    p.add_argument("--encoder-checkpoint", type=Path, help="strict native encoder state dict")
    p.add_argument("--prior", type=Path, help="JSON prior; default: estimate from training masks")
    a = p.parse_args()
    if min(a.iters, a.batch_size, a.log_interval, a.ckpt_interval) < 1:
        p.error("iterations, batch size and intervals must be positive")
    if a.workers < 0 or a.crop_size < 32 or a.lr <= 0 or a.grad_clip < 0:
        p.error("invalid worker, crop, learning-rate or clipping setting")
    if a.resume and (a.encoder_checkpoint or a.prior):
        p.error("resume restores encoder and prior; initialization overrides cannot be combined")
    return a


def main():
    a = parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    device = resolve_device(a.device)
    if device.type == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", device.index or 0)))
        torch.cuda.set_device(device)
    if a.amp and device.type != "cuda":
        raise ValueError("--amp requires CUDA")
    if a.sync_bn and (world == 1 or device.type != "cuda"):
        raise ValueError("--sync-bn requires multi-GPU torchrun")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        train(a, device, rank, world)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def train(a, device, rank, world):
    seed_everything(a.seed + rank, a.deterministic)
    work = a.work_dir or Path("work_dirs") / f"{a.model}_{a.dataset}"
    work.mkdir(parents=True, exist_ok=True)
    # Check before rank 0 creates artifacts; synchronize to avoid a startup race.
    occupied = (work / "run.json").exists() or bool(list(work.glob("iter_*.pth")))
    if world > 1:
        flag = torch.tensor(int(occupied), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        occupied = bool(flag.item())
    if not a.resume and occupied:
        raise FileExistsError("work directory contains a run; use --resume or a new --work-dir")
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("hdrnet.train")
    if rank == 0:
        log.addHandler(logging.FileHandler(work / "train.log", encoding="utf-8"))
    dataset = make_dataset(a.dataset, a.data_root, True, a.model == "da-segformer", a.crop_size)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=a.seed, shuffle=True)
    generator = torch.Generator()
    loader = DataLoader(dataset, batch_size=a.batch_size, sampler=sampler, num_workers=a.workers,
                        drop_last=True, pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                        generator=generator)
    if not len(loader):
        raise ValueError("dataset is smaller than per-process batch; lower --batch-size")
    model = build_model(a.model, a.dataset).to(device)
    if a.encoder_checkpoint:
        load_checkpoint(a.encoder_checkpoint, model.encoder)
    if a.model == "hdrnet" and not a.resume:
        prior = torch.tensor(json.loads(a.prior.read_text())["prior"]) if a.prior else \
            estimate_prior((mask for _, mask in dataset.pairs), len(DATASETS[a.dataset]["classes"]))
        model.drca.set_prior(prior)
        if rank == 0:
            write_json(work / ("resume_prior.json" if a.resume else "prior.json"),
                       {"dataset": a.dataset, "split": "train", "prior": prior.tolist()})
    if a.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    criterion = build_criterion(a.model, a.dataset).to(device)
    optimizer = AdamW(model.parameters(), lr=a.lr, betas=PROTOCOL["betas"], weight_decay=PROTOCOL["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp) if hasattr(torch.amp, "GradScaler") else \
        torch.cuda.amp.GradScaler(enabled=a.amp)
    start = epoch = offset = 0
    state = None
    if a.resume:
        state = load_checkpoint(a.resume, model, a.model, a.dataset)
        if state.get("schema_version") != 2:
            raise ValueError("resume requires a v0.2 checkpoint; legacy weights support inference only")
        for key in ("seed", "batch_size", "crop_size", "workers", "amp", "sync_bn", "lr", "iters", "deterministic"):
            if state["args"][key] != getattr(a, key):
                raise ValueError(f"resume setting differs from checkpoint: {key}")
        if state["world_size"] != world:
            raise ValueError("resume requires the same number of processes")
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, epoch, offset = state["step"], state["epoch"], state["offset"]
        if start >= a.iters:
            raise ValueError("checkpoint has already completed this schedule")
    if world > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True)
    raw = model.module if hasattr(model, "module") else model
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    if rank == 0:
        write_json(work / ("resume.json" if a.resume else "run.json"),
                   {"args": config, "protocol": PROTOCOL, "environment": environment(),
                    "world_size": world, "effective_batch_size": a.batch_size * world,
                    "classes": DATASETS[a.dataset]["classes"]})
        log.info("%s/%s: %d images, effective batch %d, device %s", a.model, a.dataset,
                 len(dataset), a.batch_size * world, device)

    def iterator(e):
        sampler.set_epoch(e)
        generator.manual_seed(a.seed + e + rank)
        return iter(loader)

    data_iter = iterator(epoch)
    for _ in range(offset):
        next(data_iter)
    if state:
        restore_rng(state["rng_states"][rank])
    model.train()
    for step in range(start, a.iters):
        if offset == len(loader):
            epoch, offset = epoch + 1, 0
            data_iter = iterator(epoch)
        batch = next(data_iter)
        offset += 1
        lr = learning_rate(step, a.lr, max_iters=a.iters, power=PROTOCOL["power"])
        set_learning_rate(optimizer, lr)
        image, mask = batch["image"].to(device), batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=a.amp):
            losses = criterion(model(image), mask)
            value = losses["loss"] if isinstance(losses, dict) else losses
        if not torch.isfinite(value):
            raise FloatingPointError(f"non-finite loss at iteration {step + 1}")
        scaler.scale(value).backward()
        if a.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        if rank == 0 and (step % a.log_interval == 0 or step + 1 == a.iters):
            record = {"step": step + 1, "lr": lr, "loss": float(value.detach())}
            if isinstance(losses, dict):
                record.update({k: float(v.detach()) for k, v in losses.items()})
            with (work / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            log.info("iter %d/%d lr %.6e loss %.4f", step + 1, a.iters, lr, record["loss"])
        if (step + 1) % a.ckpt_interval == 0 or step + 1 == a.iters:
            rng = [capture_rng()]
            if world > 1:
                rng = [None] * world
                dist.all_gather_object(rng, capture_rng())
            if rank == 0:
                path = work / f"iter_{step + 1}.pth"
                save_checkpoint(path, {"schema_version": 2, "model": raw.state_dict(),
                    "model_name": a.model, "dataset": a.dataset, "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "step": step + 1, "epoch": epoch, "offset": offset,
                    "rng_states": rng, "world_size": world, "args": config})
                log.info("saved %s", path)


if __name__ == "__main__":
    main()
