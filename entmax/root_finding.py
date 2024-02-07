"""
Bisection implementation of alpha-entmax (Peters et al., 2019).
Backward pass wrt alpha per (Correia et al., 2019). See
https://arxiv.org/pdf/1905.05702 for detailed description.
"""
# Author: Goncalo M Correia <goncalommac@gmail.com>
# Author: Ben Peters <benzurdopeters@gmail.com>
# Author: Vlad Niculae <vlad@vene.ro>
# Author: Andre Martins <andre.t.martins@gmail.com>

import torch
import torch.nn as nn
from torch.autograd import Function


class EntmaxBisectFunction(Function):
    @classmethod
    def _gp(cls, x, alpha):
        return x ** (alpha - 1)

    @classmethod
    def _gp_inv(cls, y, alpha):
        return y ** (1 / (alpha - 1))

    @classmethod
    def _p(cls, X, alpha):
        return cls._gp_inv(torch.clamp(X, min=0), alpha)

    @classmethod
    def forward(cls, ctx, X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):

        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=X.dtype, device=X.device)

        alpha_shape = list(X.shape)
        alpha_shape[dim] = 1
        alpha = alpha.expand(*alpha_shape)

        ctx.alpha = alpha
        ctx.dim = dim
        d = X.shape[dim]

        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X * (alpha - 1)
        max_val = max_val * (alpha - 1)

        # Note: when alpha < 1, tau_lo > tau_hi. This still works since dm < 0.
        tau_lo = max_val - cls._gp(1, alpha)
        tau_hi = max_val - cls._gp(1 / d, alpha)

        # Note: f_lo should always be non-negative.
        # f_lo = cls._p(X - tau_lo, alpha).sum(dim) - 1

        dm = tau_hi - tau_lo

        for it in range(n_iter):

            dm /= 2
            tau_m = tau_lo + dm
            p_m = cls._p(X - tau_m, alpha)
            f_m = p_m.sum(dim) - 1

            mask = (f_m >= 0).unsqueeze(dim)
            tau_lo = torch.where(mask, tau_m, tau_lo)

        if ensure_sum_one:
            p_m /= p_m.sum(dim=dim).unsqueeze(dim=dim)

        ctx.save_for_backward(p_m)

        return p_m

    @classmethod
    def backward(cls, ctx, dY):
        Y, = ctx.saved_tensors

        gppr = torch.where(Y > 0, Y ** (2 - ctx.alpha), Y.new_zeros(1))

        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr

        d_alpha = None
        if ctx.needs_input_grad[1]:

            # alpha gradient computation
            # d_alpha = (partial_y / partial_alpha) * dY
            # NOTE: ensure alpha is not close to 1
            # since there is an indetermination
            # batch_size, _ = dY.shape

            # shannon terms
            S = torch.where(Y > 0, Y * torch.log(Y), Y.new_zeros(1))
            # shannon entropy
            ent = S.sum(ctx.dim).unsqueeze(ctx.dim)
            Y_skewed = gppr / gppr.sum(ctx.dim).unsqueeze(ctx.dim)

            d_alpha = dY * (Y - Y_skewed) / ((ctx.alpha - 1) ** 2)
            d_alpha -= dY * (S - Y_skewed * ent) / (ctx.alpha - 1)
            d_alpha = d_alpha.sum(ctx.dim).unsqueeze(ctx.dim)

        return dX, d_alpha, None, None, None


# slightly more efficient special case for sparsemax
class SparsemaxBisectFunction(EntmaxBisectFunction):
    @classmethod
    def _gp(cls, x, alpha):
        return x

    @classmethod
    def _gp_inv(cls, y, alpha):
        return y

    @classmethod
    def _p(cls, x, alpha):
        return torch.clamp(x, min=0)

    @classmethod
    def forward(cls, ctx, X, dim=-1, n_iter=50, ensure_sum_one=True):
        return super().forward(
            ctx, X, alpha=2, dim=dim, n_iter=50, ensure_sum_one=True
        )

    @classmethod
    def backward(cls, ctx, dY):
        Y, = ctx.saved_tensors
        gppr = (Y > 0).to(dtype=dY.dtype)
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None, None, None


class NormmaxBisectFunction(Function):
    @classmethod
    def _gp(cls, x, alpha):
        return x ** (alpha - 1)

    @classmethod
    def _gp_inv(cls, y, alpha):
        return y ** (1 / (alpha - 1))

    @classmethod
    def _p(cls, X, alpha):
        return cls._gp_inv(torch.clamp(X, min=0), alpha)

    @classmethod
    def forward(cls, ctx, X, alpha=2, dim=-1, n_iter=50):

        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=X.dtype, device=X.device)

        alpha_shape = list(X.shape)
        alpha_shape[dim] = 1
        alpha = alpha.expand(*alpha_shape)

        ctx.alpha = alpha
        ctx.dim = dim
        d = X.shape[dim]

        max_val, _ = X.max(dim=dim, keepdim=True)
        tau_lo = max_val - cls._gp(1, alpha)  # 1
        tau_hi = max_val - cls._gp(1 / d, alpha)  # (1/d)**(alpha-1)

        # f_lo = (cls._p(X - tau_lo, alpha) ** alpha).sum(dim) - 1
        dm = tau_hi - tau_lo

        for it in range(n_iter):
            dm /= 2
            tau_m = tau_lo + dm
            p_m = cls._p(X - tau_m, alpha)  # [X - tau]_+ ** (1/(alpha-1))
            f_m = (p_m ** alpha).sum(dim) - 1
            mask = (f_m >= 0).unsqueeze(dim)
            tau_lo = torch.where(mask, tau_m, tau_lo)

        p_m /= p_m.sum(dim=dim).unsqueeze(dim=dim)

        ctx.save_for_backward(p_m)

        return p_m

    @classmethod
    def backward(cls, ctx, dY):
        Y, = ctx.saved_tensors

        a = torch.where(Y > 0, Y, Y.new_zeros(1))
        b = torch.where(Y > 0, Y ** (2 - ctx.alpha), Y.new_zeros(1))

        dX = dY * b
        q = dX.sum(ctx.dim).unsqueeze(ctx.dim)
        dX -= q * a
        q = (dY * a).sum(ctx.dim).unsqueeze(ctx.dim)
        dX -= q * (b - b.sum(ctx.dim).unsqueeze(ctx.dim) * a)
        dX *= ((a ** ctx.alpha).sum(ctx.dim).unsqueeze(ctx.dim) **
               ((ctx.alpha - 1) / ctx.alpha))
        dX /= (ctx.alpha - 1)

        return dX, None, None, None, None


class BudgetBisectFunction(Function):
    @classmethod
    def forward(cls, ctx, X, budget=2, dim=-1, n_iter=50):

        if not isinstance(budget, torch.Tensor):
            budget = torch.tensor(budget, dtype=X.dtype, device=X.device)

        budget_shape = list(X.shape)
        budget_shape[dim] = 1
        budget = budget.expand(*budget_shape)

        ctx.budget = budget
        ctx.dim = dim

        max_val, _ = X.max(dim=dim, keepdim=True)
        min_val, _ = X.min(dim=dim, keepdim=True)

        # If tau = max_val - B/N, X-tau <= B/N, hence sum(max(0, X-tau)) <= B.
        # If tau = min_val - B/N, X-tau >= B/N, hence sum(min(1, X-tau)) >= B.
        tau_lo = min_val - budget / X.shape[dim]
        tau_hi = max_val - budget / X.shape[dim]

        # f_lo = (torch.clamp(X - tau_lo, min=0, max=1).sum(dim).unsqueeze(dim)
        #         - budget)
        dm = tau_hi - tau_lo

        for it in range(n_iter):
            dm /= 2
            tau_m = tau_lo + dm
            p_m = torch.clamp(X - tau_m, min=0, max=1)
            f_m = p_m.sum(dim).unsqueeze(dim) - budget
            mask = (f_m >= 0)
            tau_lo = torch.where(mask, tau_m, tau_lo)

        ctx.save_for_backward(p_m)

        return p_m

    @classmethod
    def backward(cls, ctx, dY):
        Y, = ctx.saved_tensors

        # tau = (size_one  + sum(score[active]) - budget) / size_active
        # dtau / dscore = 1 / size_active
        # J = Diag(s) - ss' / size_active, s indicator vector of active indices.

        active = (Y > 0) & (Y < 1)
        active_size = active.sum(ctx.dim)
        v_hat = ((dY * active).sum(ctx.dim) /
                 active_size.to(Y.dtype).squeeze(ctx.dim)).unsqueeze(ctx.dim)
        dX = torch.where((Y > 0) & (Y < 1), dY - v_hat, Y.new_zeros(1))

        return dX, None, None, None, None


def entmax_bisect(X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):
    """alpha-entmax: normalizing sparse transform (a la softmax).

    Solves the optimization problem:

        max_p <x, p> - H_a(p)    s.t.    p >= 0, sum(p) == 1.

    where H_a(p) is the Tsallis alpha-entropy with custom alpha >= 1,
    using a bisection (root finding, binary search) algorithm.

    This function is differentiable with respect to both X and alpha.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor.

    alpha : float or torch.Tensor
        Tensor of alpha parameters (> 1) to use. If scalar
        or python float, the same value is used for all rows, otherwise,
        it must have shape (or be expandable to)
        alpha.shape[j] == (X.shape[j] if j != dim else 1)
        A value of alpha=2 corresponds to sparsemax, and alpha=1 would in theory recover
        softmax. For numeric reasons, this algorithm does not work with `alpha=1`: if you
        want softmax, we recommend `torch.nn.softmax`.

    dim : int
        The dimension along which to apply alpha-entmax.

    n_iter : int
        Number of bisection iterations. For float32, 24 iterations should
        suffice for machine precision.

    ensure_sum_one : bool,
        Whether to divide the result by its sum. If false, the result might
        sum to close but not exactly 1, which might cause downstream problems.

    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    """
    return EntmaxBisectFunction.apply(X, alpha, dim, n_iter, ensure_sum_one)


def sparsemax_bisect(X, dim=-1, n_iter=50, ensure_sum_one=True):
    """sparsemax: normalizing sparse transform (a la softmax), via bisection.

    Solves the projection:

        min_p ||x - p||_2   s.t.    p >= 0, sum(p) == 1.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor.

    dim : int
        The dimension along which to apply sparsemax.

    n_iter : int
        Number of bisection iterations. For float32, 24 iterations should
        suffice for machine precision.

    ensure_sum_one : bool,
        Whether to divide the result by its sum. If false, the result might
        sum to close but not exactly 1, which might cause downstream problems.

    Note: This function does not yet support normalizing along anything except
    the last dimension. Please use transposing and views to achieve more
    general behavior.

    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    """
    return SparsemaxBisectFunction.apply(X, dim, n_iter, ensure_sum_one)


def normmax_bisect(X, alpha=2, dim=-1, n_iter=50):
    """alpha-normmax: normalizing sparse transform (a la softmax and entmax).

    Solves the optimization problem:

        max_p <x, p> - ||p||_alpha    s.t.    p >= 0, sum(p) == 1.

    where ||.||_alpha is the alpha-norm, with custom alpha >= 1,
    using a bisection (root finding, binary search) algorithm.

    This function is differentiable with respect to X.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor.

    alpha : float or torch.Tensor
        Tensor of alpha parameters (> 1) to use. If scalar
        or python float, the same value is used for all rows, otherwise,
        it must have shape (or be expandable to)
        alpha.shape[j] == (X.shape[j] if j != dim else 1).

    dim : int
        The dimension along which to apply alpha-normmax.

    n_iter : int
        Number of bisection iterations. For float32, 24 iterations should
        suffice for machine precision.

    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    """
    return NormmaxBisectFunction.apply(X, alpha, dim, n_iter)


def budget_bisect(X, budget=2, dim=-1, n_iter=50):
    """budget-sparsemax: normalizing sparse transform with budget (k-subsets)
    constraint.

    Solves the optimization problem:

        max_p <x, p> - 1/2 ||p||**2    s.t.    0 <= p <= 1, sum(p) == budget

    using a bisection (root finding, binary search) algorithm.

    This function is differentiable with respect to X.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor.

    budget : float or torch.Tensor
        Tensor of budget parameters (> 1) to use. If scalar
        or python float, the same value is used for all rows, otherwise,
        it must have shape (or be expandable to)
        budget.shape[j] == (X.shape[j] if j != dim else 1).

    dim : int
        The dimension along which to apply budget-sparsemax.

    n_iter : int
        Number of bisection iterations. For float32, 24 iterations should
        suffice for machine precision.

    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == budget elementwise.
    """
    return BudgetBisectFunction.apply(X, budget, dim, n_iter)


class SparsemaxBisect(nn.Module):
    def __init__(self, dim=-1, n_iter=None):
        """sparsemax: normalizing sparse transform (a la softmax) via bisection

        Solves the projection:

            min_p ||x - p||_2   s.t.    p >= 0, sum(p) == 1.

        Parameters
        ----------
        dim : int
            The dimension along which to apply sparsemax.

        n_iter : int
            Number of bisection iterations. For float32, 24 iterations should
            suffice for machine precision.
        """
        self.dim = dim
        self.n_iter = n_iter
        super().__init__()

    def forward(self, X):
        return sparsemax_bisect(X, dim=self.dim, n_iter=self.n_iter)


class EntmaxBisect(nn.Module):
    def __init__(self, alpha=1.5, dim=-1, n_iter=50):
        """alpha-entmax: normalizing sparse map (a la softmax) via bisection.

        Solves the optimization problem:

            max_p <x, p> - H_a(p)    s.t.    p >= 0, sum(p) == 1.

        where H_a(p) is the Tsallis alpha-entropy with custom alpha >= 1,
        using a bisection (root finding, binary search) algorithm.

        Parameters
        ----------
        alpha : float or torch.Tensor
            Tensor of alpha parameters (> 1) to use. If scalar
            or python float, the same value is used for all rows, otherwise,
            it must have shape (or be expandable to)
            alpha.shape[j] == (X.shape[j] if j != dim else 1)
            A value of alpha=2 corresponds to sparsemax; and alpha=1 would in theory recover
            softmax. For numeric reasons, this algorithm does not work with `alpha=1`; if you
            want softmax, we recommend `torch.nn.softmax`.

        dim : int
            The dimension along which to apply alpha-entmax.

        n_iter : int
            Number of bisection iterations. For float32, 24 iterations should
            suffice for machine precision.

        """
        self.dim = dim
        self.n_iter = n_iter
        self.alpha = alpha
        super().__init__()

    def forward(self, X):
        return entmax_bisect(
            X, alpha=self.alpha, dim=self.dim, n_iter=self.n_iter
        )


class NormmaxBisect(nn.Module):
    def __init__(self, alpha=2, dim=-1, n_iter=50):
        """alpha-normmax: normalizing sparse map (a la softmax and entmax)
        via bisection.

        Solves the optimization problem:

            max_p <x, p> - ||p||_alpha    s.t.    p >= 0, sum(p) == 1.

        where ||.||_alpha is the alpha-norm, with custom alpha >= 1,
        using a bisection (root finding, binary search) algorithm.

        Parameters
        ----------
        alpha : float or torch.Tensor
            Tensor of alpha parameters (> 1) to use. If scalar
            or python float, the same value is used for all rows, otherwise,
            it must have shape (or be expandable to)
            alpha.shape[j] == (X.shape[j] if j != dim else 1).

        dim : int
            The dimension along which to apply alpha-normmax.

        n_iter : int
            Number of bisection iterations. For float32, 24 iterations should
            suffice for machine precision.
        """
        self.dim = dim
        self.n_iter = n_iter
        self.alpha = alpha
        super().__init__()

    def forward(self, X):
        return normmax_bisect(
            X, alpha=self.alpha, dim=self.dim, n_iter=self.n_iter
        )


class BudgetBisect(nn.Module):
    def __init__(self, budget=2, dim=-1, n_iter=50):
        """budget-sparsemax: normalizing sparse transform with budget
        (k-subsets) constraint.

        Solves the optimization problem:

            max_p <x, p> - 1/2 ||p||**2    s.t.    0 <= p <= 1, sum(p) == budget

        using a bisection (root finding, binary search) algorithm.

        This function is differentiable with respect to X.

        Parameters
        ----------
        X : torch.Tensor
            The input tensor.

        budget : float or torch.Tensor
            Tensor of budget parameters (> 1) to use. If scalar
            or python float, the same value is used for all rows, otherwise,
            it must have shape (or be expandable to)
            budget.shape[j] == (X.shape[j] if j != dim else 1).

        dim : int
            The dimension along which to apply budget-sparsemax.

        n_iter : int
            Number of bisection iterations. For float32, 24 iterations should
            suffice for machine precision.
        """
        self.dim = dim
        self.n_iter = n_iter
        self.budget = budget
        super().__init__()

    def forward(self, X):
        return budget_bisect(
            X, budget=self.budget, dim=self.dim, n_iter=self.n_iter
        )
