"""Validate every image/mask pair before starting a training run."""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.datasets import DATASETS
from hdrnet.data.dataset import DisasterDataset
from hdrnet.runtime import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("outputs/data_report.json"))
    a = p.parse_args()
    spec = DATASETS[a.dataset]
    report = {"dataset": a.dataset, "classes": spec["classes"], "splits": {}}
    for name, split in spec["splits"].items():
        dataset = DisasterDataset(a.data_root, split, num_classes=len(spec["classes"]))
        counts = torch.zeros(len(spec["classes"]), dtype=torch.int64)
        for sample in dataset:
            mask = sample["mask"]
            counts += torch.bincount(mask[mask != 255], minlength=len(counts))
        report["splits"][name] = {"images": len(dataset), "pixel_counts": counts.tolist()}
    write_json(a.output, report)
    print(f"all pairs valid; saved {a.output}")


if __name__ == "__main__":
    main()
