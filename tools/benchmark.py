"""Measure model-only inference latency under an explicit runtime protocol."""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.datasets import DATASETS
from hdrnet.builder import MODELS, build_model
from hdrnet.runtime import environment, load_checkpoint, resolve_device, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=sorted(MODELS), default="hdrnet")
    p.add_argument("--dataset", choices=tuple(DATASETS), default="floodnet")
    p.add_argument("--device", default="cuda")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, default=Path("outputs/benchmark.json"))
    a = p.parse_args()
    if min(a.batch_size, a.repeats) < 1 or a.warmup < 0 or a.size < 32:
        p.error("invalid input size, batch or timing counts")
    device = resolve_device(a.device)
    model = build_model(a.model, a.dataset).to(device).eval()
    if a.checkpoint:
        load_checkpoint(a.checkpoint, model, a.model, a.dataset)
    image = torch.randn(a.batch_size, 3, a.size, a.size, device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(a.warmup):
            model(image)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(a.repeats):
            sync()
            start = time.perf_counter()
            model(image)
            sync()
            times.append((time.perf_counter() - start) * 1000)
    mean = statistics.mean(times)
    report = {"model": a.model, "dataset": a.dataset, "environment": environment(), "device": str(device),
        "input": list(image.shape), "dtype": "float32", "warmup": a.warmup, "repeats": a.repeats,
        "parameters": sum(x.numel() for x in model.parameters()), "batch_latency_ms": mean,
        "median_latency_ms": statistics.median(times), "latency_std_ms": statistics.pstdev(times),
        "images_per_second": 1000 * a.batch_size / mean,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "scope": "model forward only; excludes preprocessing, I/O and sliding-window aggregation"}
    write_json(a.output, report)
    print(f"{mean:.2f} ms/batch, {report['images_per_second']:.2f} images/s; saved {a.output}")


if __name__ == "__main__":
    main()
