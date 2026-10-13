import pytest
import torch

from configs.datasets import state_groups
from hdrnet.builder import MODELS, PROTOTYPE_MODELS, build_criterion, build_model
from hdrnet.drca import DRCA
from hdrnet.dsbr import DSBR, morphological_boundary
from hdrnet.hdad import StateHead


def test_ordinal_probabilities_depend_on_input_and_backpropagate():
    head = StateHead(4, 4, True)
    image = torch.randn(2, 4, 3, 3, requires_grad=True)
    logp, logits = head(image)
    probabilities = logp.softmax(1)
    assert torch.allclose(probabilities.sum(1), torch.ones(2, 3, 3))
    assert (logits[:, :-1] >= logits[:, 1:]).all()
    assert not torch.allclose(probabilities[0, :, 0, 0], probabilities[0, :, 0, 1])
    logp[:, 0].mean().backward()
    assert image.grad.abs().sum() > 0


def test_state_groups_match_manuscript_factorization():
    assert len(state_groups("floodnet")) == 2
    assert len(state_groups("rescuenet")) == 2
    assert state_groups("rescuenet")["2"]["ordered"]
    assert len(state_groups("fwisd")) == 4


def test_prior_validation():
    module = DRCA(16, 3, 8)
    with pytest.raises(ValueError):
        module.set_prior(torch.ones(2, 2))
    module.set_prior(torch.eye(3))
    assert module.prior.sum() == 0


def test_boundary_returns_logits_and_masks_ignore():
    module = DSBR(8, 8, 8)
    with torch.no_grad():
        module.boundary.weight.zero_()
        module.boundary.bias.fill_(-5)
    _, logits = module(torch.randn(2, 8, 4, 4), torch.randn(2, 8, 2, 2), torch.randn(2, 8, 4, 4))
    assert (logits == -5).all()
    label = torch.tensor([[[1, 255], [0, 0]]])
    assert morphological_boundary(label, (1,))[0, 0, 1] == 0


@pytest.mark.parametrize("dataset", ["floodnet", "rescuenet", "fwisd"])
def test_full_forward_backward(dataset):
    model = build_model("hdrnet", dataset).train()
    loss_fn = build_criterion("hdrnet", dataset)
    image = torch.randn(2, 3, 32, 32)
    output = model(image)
    label = torch.randint(model.drca.num_classes, (2, 32, 32))
    losses = loss_fn(output, label)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.hdad.semantic.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_all_ignored_and_absent_state_pixels_are_finite():
    model = build_model("hdrnet", "rescuenet").train()
    criterion = build_criterion("hdrnet", "rescuenet")
    for value in (0, 255):
        losses = criterion(model(torch.randn(2, 3, 32, 32)), torch.full((2, 32, 32), value))
        assert torch.isfinite(losses["loss"])
        assert losses["state"] == 0


@pytest.mark.parametrize("name", sorted(set(MODELS) - {"hdrnet"}))
def test_comparison_adapters_have_executable_interfaces(name):
    if name in PROTOTYPE_MODELS:
        with pytest.warns(UserWarning, match="prototype"):
            model = build_model(name, "floodnet")
    else:
        model = build_model(name, "floodnet")
    model.eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 10, 64, 64)
    assert torch.isfinite(output).all()
