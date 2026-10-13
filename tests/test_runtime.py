import json

import pytest
import torch

from hdrnet.engine.infer import slide_inference
from hdrnet.runtime import capture_rng, load_checkpoint, restore_rng, save_checkpoint, seed_everything, write_json


class IdentitySegmenter(torch.nn.Module):
    def forward(self, image):
        return image[:, :2]


@pytest.mark.parametrize("shape", [(20, 21), (20, 85), (85, 20), (65, 83)])
def test_sliding_windows_cover_each_pixel(shape):
    image = torch.randn(1, 3, *shape)
    result = slide_inference(IdentitySegmenter(), image, 2, 32, 21)
    assert torch.allclose(result, image[:, :2], atol=1e-6)


def test_stride_gap_is_rejected():
    with pytest.raises(ValueError):
        slide_inference(IdentitySegmenter(), torch.ones(1, 3, 50, 50), 2, 32, 40)


def test_checkpoint_round_trip_and_metadata_guard(tmp_path):
    model = torch.nn.Linear(2, 2)
    seed_everything(42)
    rng = capture_rng()
    expected = torch.rand(3)
    restore_rng(rng)
    assert torch.equal(expected, torch.rand(3))
    path = tmp_path / "checkpoint.pth"
    save_checkpoint(path, {"model": model.state_dict(), "model_name": "unit", "rng": rng})
    state = load_checkpoint(path, torch.nn.Linear(2, 2), model_name="unit")
    assert state["rng"]["numpy"]
    with pytest.raises(ValueError):
        load_checkpoint(path, model, model_name="wrong")


def test_json_rejects_nan_without_corrupting_existing_artifact(tmp_path):
    path = tmp_path / "report.json"
    write_json(path, {"valid": True})
    with pytest.raises(ValueError):
        write_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"valid": True}
