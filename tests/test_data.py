import numpy as np
import pytest
from PIL import Image

from hdrnet.data.dataset import DisasterDataset, _pairs
from hdrnet.data.prior import estimate_prior
from hdrnet.data.transforms import TrainTransform, _resize

SPLIT = {"img": "images", "ann": "masks", "img_suffix": ".png", "ann_suffix": ".png"}


def sample(root, mask):
    (root / "images").mkdir(exist_ok=True)
    (root / "masks").mkdir(exist_ok=True)
    Image.fromarray(np.zeros((*mask.shape[:2], 3), dtype=np.uint8)).save(root / "images/one.png")
    Image.fromarray(mask).save(root / "masks/one.png")


def test_pairs_fail_on_missing_mask(tmp_path):
    sample(tmp_path, np.zeros((4, 4), dtype=np.uint8))
    (tmp_path / "masks/one.png").unlink()
    with pytest.raises(FileNotFoundError, match="missing mask"):
        _pairs(tmp_path, SPLIT)


@pytest.mark.parametrize("mask", [np.zeros((4, 4, 3), dtype=np.uint8), np.full((4, 4), 9, dtype=np.uint8)])
def test_masks_validate_ontology(tmp_path, mask):
    sample(tmp_path, mask)
    with pytest.raises(ValueError):
        DisasterDataset(tmp_path, SPLIT, num_classes=2)[0]


def test_resize_preserves_singleton_mask_dimension():
    image, mask = _resize(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2)), 1, 3)
    assert image.shape == (1, 3, 3)
    assert mask.shape == (1, 3)


def test_both_flip_axes_and_rotation_preserve_labels(monkeypatch):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.arange(16).reshape(4, 4)
    transform = TrainTransform(4, scale=4, ratio_range=(1, 1), horizontal_prob=1, vertical_prob=1, rotation_prob=0)
    monkeypatch.setattr(transform, "_photometric", lambda x: x)
    _, output = transform(image, mask)
    np.testing.assert_array_equal(output, mask[::-1, ::-1])
    transform.horizontal_prob = transform.vertical_prob = 0
    transform.rotation_prob = 1
    monkeypatch.setattr(np.random, "uniform", lambda low, high: 1 if low == 1 else 90)
    _, output = transform(image, mask)
    np.testing.assert_array_equal(output, np.rot90(mask))


def test_prior_uses_image_presence_not_pixel_counts(tmp_path):
    paths = []
    for i, values in enumerate(([0, 1], [1, 2], [1, 255])):
        path = tmp_path / f"{i}.png"
        Image.fromarray(np.array([values], dtype=np.uint8)).save(path)
        paths.append(path)
    prior = estimate_prior(paths, 3)
    assert prior[0, 1] == 1
    assert prior[1, 0] == pytest.approx(1 / 3)
    assert prior.diag().sum() == 0
