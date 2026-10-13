"""Two-process CPU training catches sampler, collective and checkpoint regressions."""
from argparse import Namespace
import json

import numpy as np
import pytest
import torch
import torch.distributed as dist
from PIL import Image

from tools.train import train


def worker(rank, root, work, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    args = Namespace(model="hdrnet", dataset="floodnet", data_root=root, work_dir=work,
        iters=1, batch_size=2, workers=0, lr=6e-5, crop_size=32, device="cpu", seed=42,
        deterministic=True, amp=False, sync_bn=False, grad_clip=0., log_interval=1,
        ckpt_interval=1, resume=None, encoder_checkpoint=None, prior=None)
    try:
        train(args, torch.device("cpu"), rank, 2)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_process_training_writes_one_complete_checkpoint(tmp_path):
    root = tmp_path / "FloodNet"
    images, masks = root / "train_only_crop1024/images", root / "train_only_crop1024/labels"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    rng = np.random.default_rng(42)
    for i in range(4):
        Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(images / f"{i}.jpg")
        Image.fromarray(rng.integers(0, 10, (32, 32), dtype=np.uint8)).save(masks / f"{i}.png")
    work = tmp_path / "run"
    rendezvous = (tmp_path / "store").as_uri()
    torch.multiprocessing.spawn(worker, args=(root, work, rendezvous), nprocs=2, join=True)
    state = torch.load(work / "iter_1.pth", weights_only=True)
    assert len(state["rng_states"]) == 2
    assert state["world_size"] == 2
    assert json.loads((work / "run.json").read_text())["effective_batch_size"] == 4
