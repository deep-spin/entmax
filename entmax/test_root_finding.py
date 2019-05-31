import pytest

import torch
from torch.autograd import gradcheck

from root_finding import sparsemax_bisect, tsallis_bisect

from activations import sparsemax, tsallis15


def test_sparsemax():
    for _ in range(10):
        x = 0.5 * torch.randn(10, 30000, dtype=torch.float32)
        p1 = sparsemax(x, 1)
        p2 = sparsemax_bisect(x)
        assert torch.sum((p1 - p2) ** 2) < 1e-7


def test_tsallis15():
    for _ in range(10):
        x = 0.5 * torch.randn(10, 30000, dtype=torch.float32)
        p1 = tsallis15(x, 1)
        p2 = tsallis_bisect(x, 1.5)
        assert torch.sum((p1 - p2) ** 2) < 1e-7


def test_sparsemax_grad():

    for _ in range(10):

        x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        gradcheck(sparsemax_bisect, (x,), eps=1e-5)


@pytest.mark.parametrize('alpha', (1.2, 1.5, 1.75, 2.25))
def test_tsallis_grad(alpha):

    for _ in range(10):

        x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        _, y = torch.max(torch.randn_like(x), dim=1)

        gradcheck(tsallis_bisect, (x, alpha), eps=1e-5)
