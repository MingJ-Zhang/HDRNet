"""Exercise the actual tools on a generated dataset, including exact CPU resume."""
import json
import sys

import numpy as np
import torch
from PIL import Image

from tools import benchmark, check_data, predict, prepare_prior, test, train


def invoke(monkeypatch, module, arguments):
    monkeypatch.setattr(sys, "argv", [module.__name__, *map(str, arguments)])
    module.main()


def test_train_resume_evaluate_predict_and_benchmark(tmp_path, monkeypatch):
    root = tmp_path / "FloodNet"
    layout = [("train_only_crop1024/images", "train_only_crop1024/labels", ".png", 4),
              ("test/test-org-img", "test/test-label-img", "_lab.png", 1)]
    rng = np.random.default_rng(123)
    for image_dir, mask_dir, suffix, count in layout:
        (root / image_dir).mkdir(parents=True)
        (root / mask_dir).mkdir(parents=True)
        for i in range(count):
            Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(root / image_dir / f"{i}.jpg")
            Image.fromarray(rng.integers(0, 10, (40, 40), dtype=np.uint8)).save(root / mask_dir / f"{i}{suffix}")
    common = ["--model", "hdrnet", "--dataset", "floodnet", "--data-root", root,
              "--iters", 3, "--batch-size", 2, "--crop-size", 32, "--workers", 0,
              "--device", "cpu", "--seed", 123, "--deterministic", "--ckpt-interval", 1, "--log-interval", 1]
    work = tmp_path / "run"
    invoke(monkeypatch, train, [*common, "--work-dir", work])
    uninterrupted = torch.load(work / "iter_3.pth", weights_only=True)["model"]
    for step in (1, 2):
        resumed = tmp_path / f"resume{step}"
        invoke(monkeypatch, train, [*common, "--work-dir", resumed, "--resume", work / f"iter_{step}.pth"])
        recovered = torch.load(resumed / "iter_3.pth", weights_only=True)["model"]
        assert all(torch.equal(uninterrupted[k], recovered[k]) for k in uninterrupted)
    assert json.loads((work / "run.json").read_text())["effective_batch_size"] == 2
    assert len((work / "metrics.jsonl").read_text().splitlines()) == 3
    checkpoint = work / "iter_3.pth"
    report = tmp_path / "evaluation.json"
    invoke(monkeypatch, test, ["--model", "hdrnet", "--dataset", "floodnet", "--data-root", root,
           "--checkpoint", checkpoint, "--device", "cpu", "--workers", 0,
           "--crop-size", 32, "--stride", 21, "--output", report, "--save-predictions", tmp_path / "masks"])
    scores = json.loads(report.read_text())["scores"]
    assert scores["evaluated_pixels"] == 1600
    assert len(scores["per_class"]) == 10
    invoke(monkeypatch, predict, ["--dataset", "floodnet", "--image", root / "test/test-org-img/0.jpg",
           "--checkpoint", checkpoint, "--device", "cpu", "--crop-size", 32, "--stride", 21,
           "--output-dir", tmp_path / "prediction"])
    with Image.open(tmp_path / "prediction/mask.png") as image:
        assert image.size == (40, 40)
    invoke(monkeypatch, benchmark, ["--device", "cpu", "--size", 32, "--warmup", 0, "--repeats", 2,
           "--output", tmp_path / "benchmark.json"])
    assert json.loads((tmp_path / "benchmark.json").read_text())["parameters"] > 0
    invoke(monkeypatch, check_data, ["--dataset", "floodnet", "--data-root", root,
           "--output", tmp_path / "data.json"])
    assert json.loads((tmp_path / "data.json").read_text())["splits"]["train"]["images"] == 4
    invoke(monkeypatch, prepare_prior, ["--dataset", "floodnet", "--data-root", root,
           "--output", tmp_path / "prior.json"])
    assert json.loads((tmp_path / "prior.json").read_text())["split"] == "train"
