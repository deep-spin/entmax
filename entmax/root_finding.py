"""
Bisection implementation of alpha-entmax (Peters et al., 2019).
Backward pass wrt alpha per (Correia et al., 2019). See
https://arxiv.org/pdf/1905.05702 for detailed description.
"""
# Author: Goncalo M Correia
# Author: Ben Peters
# Author: Vlad Niculae <vlad@vene.ro>

import torch
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
    def forward(cls, ctx, X, alpha=1.5, n_iter=50, ensure_sum_one=True):

        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=X.dtype)
        alpha = alpha.unsqueeze(-1)  # different alpha per row
        ctx.alpha = alpha
        ctx.dim = dim = 1
        d = X.shape[dim]

        X = X * (alpha - 1)

        max_val, _ = X.max(dim=dim, keepdim=True)

        tau_lo = max_val - cls._gp(1, alpha)
        tau_hi = max_val - cls._gp(1 / d, alpha)

        f_lo = cls._p(X - tau_lo, alpha).sum(dim) - 1

        dm = tau_hi - tau_lo

        for it in range(n_iter):

            dm /= 2
            tau_m = tau_lo + dm
            p_m = cls._p(X - tau_m, alpha)
            f_m = p_m.sum(dim) - 1

            mask = (f_m * f_lo >= 0).unsqueeze(dim)
            tau_lo = torch.where(mask, tau_m, tau_lo)

        if ensure_sum_one:
            p_m /= p_m.sum(dim=1).unsqueeze(dim=1)

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
            batch_size, _ = dY.shape

            # shannon terms
            S = torch.where(Y > 0, Y * torch.log(Y), Y.new_zeros(1))
            # shannon entropy
            ent = S.sum(ctx.dim).unsqueeze(-1)
            Y_skewed = gppr / gppr.sum(ctx.dim).unsqueeze(-1)
            d_alpha = dY * (Y - Y_skewed) / ((ctx.alpha - 1) ** 2)
            d_alpha -= dY * (S - Y_skewed * ent) / (ctx.alpha - 1)
            d_alpha = d_alpha.sum(dim=-1)

        return dX, d_alpha, None, None


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
    def forward(cls, ctx, X, n_iter=50, ensure_sum_one=True):
        return super().forward(ctx, X, alpha=2, n_iter=50, ensure_sum_one=True)

    @staticmethod
    def backward(ctx, dY):
        Y, = ctx.saved_tensors
        gppr = (Y > 0).to(dtype=dY.dtype)
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None, None


def entmax_bisect(X, alpha=1.5, n_iter=50, ensure_sum_one=True):
    """alpha-entmax: normalizing sparse transform (a la softmax).

    Solves the optimization problem:

        max_p <x, p> - H_a(p)    s.t.    p >= 0, sum(p) == 1.

    where H_a(p) is the Tsallis alpha-entropy with custom alpha >= 1,
    using a bisection (root finding, binary search) algorithm.

    This function is differentiable with respect to both X and alpha.

    Note: This function does not yet support normalizing along anything except
    the last dimension. Please use transposing and views to achieve more general
    behavior.

    Parameters
    ----------
    X : torch.Tensor
        The input 2D tensor, to be normalized along the last dimension.

    alpha : float or torch.Tensor
        Tensor of alpha parameters (> 1) to use for each row of X. If scalar
        or python float, the same value is used for all rows. A value of
        alpha=2 corresponds to sparsemax, and alpha=1 corresponds to softmax
        (but computing it this way is likely unstable).

    n_iter : int
        Number of bisection iterations. For float32, 24 iterations should
        suffice for machine precision.

    ensure_sum_one : bool,
        Whether to divide the result by its sum. If false, the result might
        sum to close but not exactly 1, which might cause downstream problems.

    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=-1) == 1 elementwise.
    """
    return EntmaxBisectFunction.apply(X, alpha, n_iter, ensure_sum_one)


def sparsemax_bisect(X, n_iter=50, ensure_sum_one=True):
    """sparsemax: normalizing sparse transform (a la softmax), via bisection.

    Solves the projection:

        min_p ||x - p||_2   s.t.    p >= 0, sum(p) == 1.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor.

    n_iter : int
        Number of bisection iterations. For float32, 24 iterations should
        suffice for machine precision.

    ensure_sum_one : bool,
        Whether to divide the result by its sum. If false, the result might
        sum to close but not exactly 1, which might cause downstream problems.

    Note: This function does not yet support normalizing along anything except
    the last dimension. Please use transposing and views to achieve more general
    behavior.

    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=-1) == 1 elementwise.
    """
    return SparsemaxBisectFunction.apply(X, n_iter, ensure_sum_one)
