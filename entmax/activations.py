"""
An implementation of entmax (Peters et al., 2019). See
https://arxiv.org/pdf/1905.05702 for detailed description.

This builds on previous work with sparsemax (Martins & Astudillo, 2016).
See https://arxiv.org/pdf/1602.02068.

By Ben Peters and Vlad Niculae
"""

import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn as nn
from entmax.root_finding import entmax_bisect, sparsemax_bisect


def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _roll_last(X, dim):
    if dim == -1:
        return X
    elif dim < 0:
        dim = X.dim() - dim

    perm = [i for i in range(X.dim()) if i != dim] + [dim]
    return X.permute(perm)


def _threshold_and_support(input, dim=0):
    """
    Sparsemax building block: compute the threshold
    Parameters:
        input: any dimension
        dim: dimension along which to apply the sparsemax
    Returns:
        the threshold value
    """
    input_srt, _ = torch.sort(input, descending=True, dim=dim)
    input_cumsum = input_srt.cumsum(dim) - 1
    rhos = _make_ix_like(input, dim)
    support = rhos * input_srt > input_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = input_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input.dtype)
    return tau, support_size


def _threshold_and_support_topk(input, dim=0, k=100):
    """
    Sparsemax building block: compute the threshold
    Parameters:
        input: any dimension
        dim: dimension along which to apply the sparsemax
    Returns:
        the threshold value
    """

    if k >= input.shape[dim]:  # do full sort
        topk, _ = torch.sort(input, dim=dim, descending=True)
    else:
        topk, _ = torch.topk(input, k=k, dim=dim)

    topk_cumsum = topk.cumsum(dim) - 1
    rhos = _make_ix_like(topk, dim)
    support = rhos * topk > topk_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = topk_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input.dtype)

    unsolved = (support_size == k).squeeze(dim)

    if torch.any(unsolved):
        in_ = _roll_last(input, dim)[unsolved]
        tau_, ss_ = _threshold_and_support_topk(in_, dim=-1, k=2 * k)
        _roll_last(tau, dim)[unsolved] = tau_
        _roll_last(support_size, dim)[unsolved] = ss_

    return tau, support_size


class SparsemaxFunction(Function):

    @staticmethod
    def forward(ctx, input, dim=0):
        """
        sparsemax: normalizing sparse transform (a la softmax)
        Parameters:
            input (Tensor): any shape
            dim: dimension along which to apply sparsemax
        Returns:
            output (Tensor): same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val  # same numerical stability trick as for softmax
        tau, supp_size = _threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None


class SparsemaxFunctionTopK(Function):

    @staticmethod
    def forward(ctx, input, dim=0, k=100):
        """
        sparsemax: normalizing sparse transform (a la softmax)
        Parameters:
            input (Tensor): any shape
            dim: dimension along which to apply sparsemax
        Returns:
            output (Tensor): same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val  # same numerical stability trick as for softmax
        tau, supp_size = _threshold_and_support_topk(input, dim=dim, k=k)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return SparsemaxFunction.backward(ctx, grad_output) + (None,)


sparsemax = SparsemaxFunction.apply
sparsemax_topk = SparsemaxFunctionTopK.apply


class Sparsemax(nn.Module):

    def __init__(self, dim=0):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)


class SparsemaxTopK(nn.Module):

    def __init__(self, dim=0, k=100):
        self.dim = dim
        self.k = k
        super(SparsemaxTopK, self).__init__()

    def forward(self, input):
        return sparsemax_topk(input, self.dim, self.k)


class LogSparsemax(nn.Module):

    def __init__(self, dim=0):
        self.dim = dim
        super(LogSparsemax, self).__init__()

    def forward(self, input):
        return torch.log(sparsemax(input, self.dim))


class LogSparsemaxTopK(nn.Module):

    def __init__(self, dim=0, k=100):
        self.dim = dim
        self.k = k
        super(LogSparsemaxTopK, self).__init__()

    def forward(self, input):
        return torch.log(sparsemax_topk(input, self.dim, self.k))


def _entmax_threshold_and_support(input, dim=0):
    Xsrt, _ = torch.sort(input, descending=True, dim=dim)

    rho = _make_ix_like(input, dim)
    mean = Xsrt.cumsum(dim) / rho
    mean_sq = (Xsrt ** 2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean ** 2)
    delta = (1 - ss) / rho

    # NOTE this is not exactly the same as in reference algo
    # Fortunately it seems the clamped values never wrongly
    # get selected by tau <= sorted_z. Prove this!
    delta_nz = torch.clamp(delta, 0)
    tau = mean - torch.sqrt(delta_nz)

    support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
    tau_star = tau.gather(dim, support_size - 1)
    return tau_star, support_size


def _entmax_threshold_and_support_topk(input, dim=0, k=100):

    if k >= input.shape[dim]:  # do full sort
        Xsrt, _ = torch.sort(input, dim=dim, descending=True)
    else:
        Xsrt, _ = torch.topk(input, k=k, dim=dim)

    rho = _make_ix_like(Xsrt, dim)
    mean = Xsrt.cumsum(dim) / rho
    mean_sq = (Xsrt ** 2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean ** 2)
    delta = (1 - ss) / rho

    # NOTE this is not exactly the same as in reference algo
    # Fortunately it seems the clamped values never wrongly
    # get selected by tau <= sorted_z. Prove this!
    delta_nz = torch.clamp(delta, 0)
    tau = mean - torch.sqrt(delta_nz)

    support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
    tau_star = tau.gather(dim, support_size - 1)

    unsolved = (support_size == k).squeeze(dim)

    if torch.any(unsolved):
        X_ = _roll_last(input, dim)[unsolved]
        tau_, ss_ = _entmax_threshold_and_support_topk(X_, dim=-1, k=2 * k)
        _roll_last(tau_star, dim)[unsolved] = tau_
        _roll_last(support_size, dim)[unsolved] = ss_

    return tau_star, support_size


class Entmax15Function(Function):
    @staticmethod
    def forward(ctx, X, dim=0):
        ctx.dim = dim

        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as for softmax
        X = X / 2  # divide by 2 to solve actual Entmax

        tau_star, _ = _entmax_threshold_and_support(X, dim)

        Y = torch.clamp(X - tau_star, min=0) ** 2
        ctx.save_for_backward(Y)
        return Y

    @staticmethod
    def backward(ctx, dY):
        Y, = ctx.saved_tensors
        gppr = Y.sqrt()  # = 1 / g'' (Y)
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None


class Entmax15TopKFunction(Entmax15Function):
    @staticmethod
    def forward(ctx, X, dim=0, k=100):
        ctx.dim = dim

        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val  # same numerical stability trick as for softmax
        X = X / 2  # divide by 2 to solve actual Entmax

        tau_star, _ = _entmax_threshold_and_support_topk(X, dim=dim, k=k)

        Y = torch.clamp(X - tau_star, min=0) ** 2
        ctx.save_for_backward(Y)
        return Y

    @staticmethod
    def backward(ctx, dY):
        return Entmax15Function.backward(ctx, dY) + (None,)


entmax15 = Entmax15Function.apply
entmax15_topk = Entmax15TopKFunction.apply


class Entmax15(nn.Module):

    def __init__(self, dim=0):
        self.dim = dim
        super(Entmax15, self).__init__()

    def forward(self, X):
        return entmax15(X, self.dim)


class LogEntmax15(nn.Module):

    def __init__(self, dim=0):
        self.dim = dim
        super(LogEntmax15, self).__init__()

    def forward(self, X):
        return torch.log(entmax15(X, self.dim))


class Entmax15TopK(nn.Module):

    def __init__(self, dim=0, k=100):
        self.dim = dim
        self.k = k
        super(Entmax15TopK, self).__init__()

    def forward(self, X):
        return entmax15_topk(X, self.dim, self.k)


class LogEntmax15TopK(nn.Module):

    def __init__(self, dim=0, k=100):
        self.dim = dim
        self.k = k
        super(LogEntmax15TopK, self).__init__()

    def forward(self, X):
        return torch.log(entmax15_topk(X, self.dim, self.k))


class LogEntmaxBisect(nn.Module):
    def __init__(self, alpha=1.5, n_iter=50):
        self.alpha = alpha
        self.n_iter = n_iter
        super(LogEntmaxBisect, self).__init__()

    def forward(self, X):
        assert X.dim() == 2

        p_star =  entmax_bisect(X, self.alpha, self.n_iter)
        p_star /= p_star.sum(dim=1).unsqueeze(dim=1)

        return torch.log(p_star)


class LogSparsemaxBisect(nn.Module):
    def __init__(self, n_iter=50):
        self.n_iter = n_iter
        super(LogSparsemaxBisect, self).__init__()

    def forward(self, X):
        assert X.dim() == 2

        p_star = sparsemax_bisect(X, self.n_iter)
        p_star /= p_star.sum(dim=1).unsqueeze(dim=1)

        return torch.log(p_star)
