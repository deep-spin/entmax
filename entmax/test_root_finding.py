import pytest
from itertools import product
from functools import partial

import torch
from torch.autograd import gradcheck

from entmax.root_finding import (sparsemax_bisect, entmax_bisect,
                                 normmax_bisect, budget_bisect)
from entmax.activations import sparsemax, entmax15


# @pytest.mark.parametrize("dim", (0, 1, 2))
# def test_dim(dim, Map):
# for _ in range(10):
# x = torch.randn(5, 6, 7, requires_grad=True, dtype=torch.float64)
# # gradcheck(f, (x,))


def test_sparsemax():
    x = 0.5 * torch.randn(4, 6, dtype=torch.float32)
    p1 = sparsemax(x, 1)
    p2 = sparsemax_bisect(x)
    assert torch.sum((p1 - p2) ** 2) < 1e-7


def test_entmax15():
    x = 0.5 * torch.randn(4, 6, dtype=torch.float32)
    p1 = entmax15(x, 1)
    p2 = entmax_bisect(x, alpha=1.5)
    assert torch.sum((p1 - p2) ** 2) < 1e-7


def test_normmax():
    x = torch.tensor([0, .25, .5, .75, 1], dtype=torch.float32)
    p1 = normmax_bisect(x, alpha=2, dim=0)
    p2 = torch.tensor([0.0000, 0.0239, 0.1746, 0.3254, 0.4761],
                      dtype=torch.float32)
    assert torch.sum((p1 - p2) ** 2) < 1e-7
    p1 = normmax_bisect(x, alpha=1000, dim=0)
    p2 = torch.tensor([0.0000, 0.0000, 0.3330, 0.3334, 0.3336],
                      dtype=torch.float32)
    assert torch.sum((p1 - p2) ** 2) < 1e-7


def test_budget():
    x = torch.tensor([0, .25, .5, .75, 1], dtype=torch.float32)
    p1 = budget_bisect(x, budget=2, dim=0)
    p2 = torch.tensor([0.0000, 0.1250, 0.3750, 0.6250, 0.8750],
                      dtype=torch.float32)
    assert torch.sum((p1 - p2) ** 2) < 1e-7
    p1 = budget_bisect(2*x, budget=3, dim=0)
    p2 = torch.tensor([0.0000, 0.2500, 0.7500, 1.0000, 1.0000],
                      dtype=torch.float32)
    assert torch.sum((p1 - p2) ** 2) < 1e-7
    for c in (1, 2, 3, 4, 5):
        p1 = budget_bisect(c*x, budget=1, dim=0)
        p2 = sparsemax(c*x, dim=0)
        assert torch.sum((p1 - p2) ** 2) < 1e-7


def test_sparsemax_grad():
    x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    gradcheck(sparsemax_bisect, (x,), eps=1e-5)


@pytest.mark.parametrize("alpha", (0.2, 0.5, 0.75, 1.2, 1.5, 1.75, 2.25))
def test_entmax_grad(alpha):
    alpha = torch.tensor(alpha, dtype=torch.float64, requires_grad=True)
    x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    gradcheck(entmax_bisect, (x, alpha), eps=1e-5)


@pytest.mark.parametrize("alpha", (1.2, 1.5, 2, 5, 10))
def test_normmax_grad(alpha):
    alpha = torch.tensor(alpha, dtype=torch.float64)
    x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    gradcheck(normmax_bisect, (x, alpha), eps=1e-5)


@pytest.mark.parametrize("k", (0, 1, 2, 3, 4, 5, 6))
def test_budget_grad(k):
    k = torch.tensor(k, dtype=torch.float64)
    x = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    gradcheck(budget_bisect, (x, k), eps=1e-5)


def test_entmax_correct_multiple_alphas():
    n = 4
    x = torch.randn(n, 6, dtype=torch.float64, requires_grad=True)
    alpha = 0.05 + 2.5*torch.rand((n, 1), dtype=torch.float64, requires_grad=True)

    p1 = entmax_bisect(x, alpha)
    p2_ = [
        entmax_bisect(x[i].unsqueeze(0), alpha[i].item()).squeeze()
        for i in range(n)
    ]
    p2 = torch.stack(p2_)

    assert torch.allclose(p1, p2)


def test_entmax_grad_multiple_alphas():

    n = 4
    x = torch.randn(n, 6, dtype=torch.float64, requires_grad=True)
    alpha = 0.05 + 2.5*torch.rand((n, 1), dtype=torch.float64, requires_grad=True)
    gradcheck(entmax_bisect, (x, alpha), eps=1e-5)


def test_normmax_grad_multiple_alphas():
    n = 4
    x = torch.randn(n, 6, dtype=torch.float64, requires_grad=True)
    alpha = 1 + 2.5*torch.rand((n, 1), dtype=torch.float64)
    gradcheck(normmax_bisect, (x, alpha), eps=1e-5)


def test_budget_grad_multiple_k():
    n = 4
    x = torch.randn(n, 6, dtype=torch.float64, requires_grad=True)
    k = 6 * torch.rand((n, 1), dtype=torch.float64)
    gradcheck(budget_bisect, (x, k), eps=1e-5)


@pytest.mark.parametrize("dim", (0, 1, 2, 3))
def test_normmax_arbitrary_dimension(dim):
    shape = [3, 4, 2, 5]
    X = torch.randn(*shape, dtype=torch.float64)

    P = normmax_bisect(X, alpha=2, dim=dim)

    ranges = [
        list(range(k)) if i != dim else [slice(None)]
        for i, k in enumerate(shape)
    ]

    for ix in product(*ranges):
        x = X[ix].unsqueeze(0)
        p_true = normmax_bisect(x, alpha=2, dim=-1)
        assert torch.allclose(P[ix], p_true)


@pytest.mark.parametrize("dim", (0, 1, 2, 3))
def test_normmax_grad_arbitrary_dimension(dim):
    shape = [3, 4, 2, 5]

    f = partial(normmax_bisect, alpha=2, dim=dim)
    X = torch.randn(*shape, dtype=torch.float64, requires_grad=True)

    gradcheck(f, (X,), eps=1e-5)


@pytest.mark.parametrize("dim", (0, 1, 2, 3))
def test_budget_arbitrary_dimension(dim):
    shape = [3, 4, 2, 5]
    X = torch.randn(*shape, dtype=torch.float64)

    P = budget_bisect(X, budget=2, dim=dim)

    ranges = [
        list(range(k)) if i != dim else [slice(None)]
        for i, k in enumerate(shape)
    ]

    for ix in product(*ranges):
        x = X[ix].unsqueeze(0)
        p_true = budget_bisect(x, budget=2, dim=-1)
        assert torch.allclose(P[ix], p_true)


@pytest.mark.parametrize("dim", (0, 1, 2, 3))
def test_budget_grad_arbitrary_dimension(dim):
    shape = [3, 4, 2, 5]

    f = partial(budget_bisect, budget=2, dim=dim)
    X = torch.randn(*shape, dtype=torch.float64, requires_grad=True)

    gradcheck(f, (X,), eps=1e-5)


@pytest.mark.parametrize("dim", (0, 1, 2, 3))
def test_arbitrary_dimension(dim):
    shape = [3, 4, 2, 5]
    X = torch.randn(*shape, dtype=torch.float64)

    alpha_shape = shape
    alpha_shape[dim] = 1

    alphas = 0.05 + 2.5*torch.rand(alpha_shape, dtype=torch.float64)

    P = entmax_bisect(X, alpha=alphas, dim=dim)

    ranges = [
        list(range(k)) if i != dim else [slice(None)]
        for i, k in enumerate(shape)
    ]

    for ix in product(*ranges):
        x = X[ix].unsqueeze(0)
        alpha = alphas[ix].item()
        p_true = entmax_bisect(x, alpha=alpha, dim=-1)
        assert torch.allclose(P[ix], p_true)


@pytest.mark.parametrize("dim", (0, 1, 2, 3))
def test_arbitrary_dimension_grad(dim):
    shape = [3, 4, 2, 5]

    alpha_shape = shape
    alpha_shape[dim] = 1

    f = partial(entmax_bisect, dim=dim)

    X = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    alphas = 0.05 + 2.5*torch.rand(
        alpha_shape, dtype=torch.float64, requires_grad=True
    )
    gradcheck(f, (X, alphas), eps=1e-5)
