# bisection
import torch
from torch.autograd import Function


def assert_equal(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def _p(x, tau):
    return torch.clamp(x - tau, min=0)


def _tsallis_gp(x, alpha):
    return x ** (alpha - 1)


def _tsallis_gp_inv(y, alpha):
    return y ** (1 / (alpha - 1))


def _tsallis_p(X,  alpha):
    return _tsallis_gp_inv(torch.clamp(X, min=0), alpha)


# TODO: support other dims other than 1. The code likely same, but needs tested
class SparsemaxBisectFunction(Function):

    @staticmethod
    def forward(ctx, X, n_iter=50, ensure_sum_one=True):

        ctx.dim = dim = 1
        d = X.shape[dim]

        max_val, _ = X.max(dim=dim, keepdim=True)

        tau_lo = max_val - 1
        tau_hi = max_val - (1 / d)

        f_lo = _p(X, tau_lo).sum(dim) - 1
        f_hi = _p(X, tau_hi).sum(dim) - 1

        dm = tau_hi - tau_lo

        for it in range(n_iter):

            dm /= 2
            tau_m = tau_lo + dm
            p_m = _p(X, tau_m)
            f_m = p_m.sum(dim) - 1

            mask = (f_m * f_lo >= 0).unsqueeze(dim)
            tau_lo = torch.where(mask, tau_m, tau_lo)

        if ensure_sum_one:
            p_m /= p_m.sum(dim=1).unsqueeze(dim=1)

        ctx.save_for_backward(p_m)

        return p_m

    @staticmethod
    def backward(ctx, dY):
        Y, = ctx.saved_tensors
        gppr = (Y > 0).to(dtype=dY.dtype)
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None


class TsallisBisectFunction(Function):

    @staticmethod
    def forward(ctx, X, alpha=1.5, n_iter=50, ensure_sum_one=True):

        ctx.alpha = alpha
        ctx.dim = dim = 1
        d = X.shape[dim]

        X = X * (alpha - 1)

        max_val, _ = X.max(dim=dim, keepdim=True)

        minv = _tsallis_gp(0, alpha)

        tau_lo = max_val - _tsallis_gp(1, alpha)
        tau_hi = max_val - _tsallis_gp(1 / d, alpha)

        f_lo = _tsallis_p(X - tau_lo, alpha).sum(dim) - 1
        f_hi = _tsallis_p(X - tau_hi, alpha).sum(dim) - 1

        dm = tau_hi - tau_lo

        for it in range(n_iter):

            dm /= 2
            tau_m = tau_lo + dm
            p_m = _tsallis_p(X - tau_m, alpha)
            f_m = p_m.sum(dim) - 1

            mask = (f_m * f_lo >= 0).unsqueeze(dim)
            tau_lo = torch.where(mask, tau_m, tau_lo)

        if ensure_sum_one:
            p_m /= p_m.sum(dim=1).unsqueeze(dim=1)

        ctx.save_for_backward(p_m)

        return p_m

    @staticmethod
    def backward(ctx, dY):
        Y, = ctx.saved_tensors

        gppr = torch.where(Y > 0, Y ** (2 - ctx.alpha), Y.new_zeros(1))

        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None, None


sparsemax_bisect = SparsemaxBisectFunction.apply
tsallis_bisect = TsallisBisectFunction.apply
