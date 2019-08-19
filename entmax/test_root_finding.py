import pytest

import torch
from torch.autograd import gradcheck

from entmax.root_finding import sparsemax_bisect, entmax_bisect

from entmax.activations import sparsemax, entmax15


def test_sparsemax():
    for _ in range(10):
        x = 0.5 * torch.randn(10, 30000, dtype=torch.float32)
        p1 = sparsemax(x, 1)
        p2 = sparsemax_bisect(x)
        assert torch.sum((p1 - p2) ** 2) < 1e-7


def test_entmax15():
    for _ in range(10):
        x = 0.5 * torch.randn(10, 30000, dtype=torch.float32)
        p1 = entmax15(x, 1)
        p2 = entmax_bisect(x, alpha=1.5)
        assert torch.sum((p1 - p2) ** 2) < 1e-7


def test_sparsemax_grad():

    for _ in range(10):

        x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        gradcheck(sparsemax_bisect, (x,), eps=1e-5)


@pytest.mark.parametrize("alpha", (1.2, 1.5, 1.75, 2.25))
def test_entmax_grad(alpha):

    alpha = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)

    for _ in range(10):
        x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        gradcheck(entmax_bisect, (x, alpha), eps=1e-5)


def test_entmax_correct_multiple_alphas():
    n = 4
    x = torch.randn(n, 6, dtype=torch.float64, requires_grad=True)
    alpha = 1.05 + torch.rand(n, dtype=torch.float64, requires_grad=True)

    p1 = entmax_bisect(x, alpha)
    p2_ = [
        entmax_bisect(x[i].unsqueeze(0), alpha[i].item()).squeeze() for i in range(n)
    ]
    p2 = torch.stack(p2_)

    assert torch.allclose(p1, p2)


def test_entmax_grad_multiple_alphas():

    for _ in range(10):
        x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        alpha = 1.05 + torch.rand(4, dtype=torch.float64, requires_grad=True)
        gradcheck(entmax_bisect, (x, alpha), eps=1e-5)
