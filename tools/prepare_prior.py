"""Export a training-only category co-occurrence prior."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.datasets import DATASETS
from hdrnet.data import make_dataset
from hdrnet.data.prior import estimate_prior
from hdrnet.runtime import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    dataset = make_dataset(a.dataset, a.data_root, train=True)
    spec = DATASETS[a.dataset]
    prior = estimate_prior((mask for _, mask in dataset.pairs), len(spec["classes"]))
    write_json(a.output, {"dataset": a.dataset, "split": "train", "images": len(dataset),
                          "classes": spec["classes"], "prior": prior.tolist()})
    print(f"saved {a.output}: {len(dataset)} training masks")


if __name__ == "__main__":
    main()
