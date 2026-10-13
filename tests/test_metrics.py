import pytest
import torch

from hdrnet.engine.metrics import ConfusionMeter


def test_confusion_and_f1_identity():
    meter = ConfusionMeter(3)
    meter.update(torch.tensor([0, 1, 1, 2, 2]), torch.tensor([0, 0, 1, 1, 255]))
    scores = meter.compute()
    assert scores["confusion_matrix"] == [[1, 1, 0], [0, 1, 1], [0, 0, 0]]
    assert scores["evaluated_pixels"] == 4
    for row in scores["per_class"]:
        assert row["F1"] == pytest.approx(200 * row["IoU"] / (100 + row["IoU"]))
    assert scores["mF1"] == pytest.approx(scores["paper_mF1"])
    assert scores["per_class"][2]["F1"] == 0  # false positives still count


def test_empty_and_absent_classes_are_json_safe():
    result = ConfusionMeter(2).compute()
    assert result["mIoU"] == 0
    assert result["per_class"][0]["IoU"] is None


@pytest.mark.parametrize("prediction,target", [([2], [0]), ([0], [3]), ([0, 1], [0])])
def test_invalid_labels_or_shapes_raise(prediction, target):
    with pytest.raises(ValueError):
        ConfusionMeter(2).update(torch.tensor(prediction), torch.tensor(target))
