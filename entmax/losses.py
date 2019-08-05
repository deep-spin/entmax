import torch
import torch.nn as nn
from torch.autograd import Function

from entmax.activations import sparsemax, entmax15
from entmax.root_finding import entmax_bisect, sparsemax_bisect


def _fy_backward(ctx, grad_output):
    p_star, = ctx.saved_tensors
    grad = grad_output.unsqueeze(1) * p_star
    return grad


# computes Omega(y_true) - Omega(p*)
def _omega_entmax(p_star, alpha):
    return (1 - (p_star ** alpha).sum(dim=1)) / (alpha * (alpha - 1))


# more efficient specializations
def _omega_entmax15(p_star):
    return (1 - (p_star * torch.sqrt(p_star)).sum(dim=1)) / 0.75


def _omega_sparsemax(p_star):
    return (1 - (p_star ** 2).sum(dim=1)) / 2


class _GenericLoss(nn.Module):

    def __init__(self, weight=None, ignore_index=-100,
                 reduction='elementwise_mean'):
        assert reduction in ['elementwise_mean', 'sum', 'none']
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index
        super(_GenericLoss, self).__init__()

    def forward(self, input, target):
        loss = self.loss(input, target)
        if self.ignore_index >= 0:
            ignored_positions = target == self.ignore_index
            size = float((target.size(0) - ignored_positions.sum()).item())
            loss.masked_fill_(ignored_positions, 0.0)
        else:
            size = float(target.size(0))
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'elementwise_mean':
            loss = loss.sum() / size
        return loss


class _GenericLossFunction(Function):
    @classmethod
    def forward(cls, ctx, input, target):
        return NotImplemented

    @classmethod
    def backward(cls, ctx, grad_output):
        return _fy_backward(ctx, grad_output), None


class SparsemaxLossFunction(_GenericLossFunction):

    @classmethod
    def forward(cls, ctx, input, target, k=None):
        """
        input (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """
        assert input.shape[0] == target.shape[0]

        p_star = sparsemax(input, 1, k)
        loss = _omega_sparsemax(p_star)

        p_star.scatter_add_(1, target.unsqueeze(1),
                            torch.full_like(p_star, -1))
        loss += torch.einsum("ij,ij->i", p_star, input)

        ctx.save_for_backward(p_star)

        # loss = torch.clamp(loss, min=0.0)  # needed?
        return loss

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_input = super(SparsemaxLossFunction, cls).backward(ctx, grad_output)
        return grad_input + (None,)


class SparsemaxBisectLossFunction(_GenericLossFunction):

    @classmethod
    def forward(cls, ctx, input, target, n_iter=50):
        """
        input (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """
        assert input.shape[0] == target.shape[0]

        p_star = sparsemax_bisect(input, n_iter)

        # this is onw done directly in sparsemax_bisect
        # p_star /= p_star.sum(dim=1).unsqueeze(dim=1)

        loss = _omega_sparsemax(p_star)

        p_star.scatter_add_(1, target.unsqueeze(1),
                            torch.full_like(p_star, -1))
        loss += torch.einsum("ij,ij->i", p_star, input)

        ctx.save_for_backward(p_star)

        # loss = torch.clamp(loss, min=0.0)  # needed?
        return loss

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_input = super(SparsemaxBisectLossFunction, cls).backward(ctx, grad_output)
        return grad_input + (None,)


class Entmax15LossFunction(_GenericLossFunction):

    @classmethod
    def forward(cls, ctx, input, target, k=None):
        """
        input (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """
        assert input.shape[0] == target.shape[0]

        p_star = entmax15(input, 1, k)
        loss = _omega_entmax15(p_star)

        p_star.scatter_add_(1, target.unsqueeze(1),
                            torch.full_like(p_star, -1))
        loss += torch.einsum("ij,ij->i", p_star, input)

        ctx.save_for_backward(p_star)

        # loss = torch.clamp(loss, min=0.0)  # needed?
        return loss

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_input = super(Entmax15LossFunction, cls).backward(ctx, grad_output)
        return grad_input + (None,)


class EntmaxBisectLossFunction(_GenericLossFunction):

    @classmethod
    def forward(cls, ctx, input, target, alpha=1.5, n_iter=50):
        """
        input (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """
        assert input.shape[0] == target.shape[0]

        p_star = entmax_bisect(input, alpha, n_iter)

        # this is now done directly in entmax_bisect
        # p_star /= p_star.sum(dim=1).unsqueeze(dim=1)

        loss = _omega_entmax(p_star, alpha)

        p_star.scatter_add_(1, target.unsqueeze(1),
                            torch.full_like(p_star, -1))
        loss += torch.einsum("ij,ij->i", p_star, input)

        ctx.save_for_backward(p_star)

        # loss = torch.clamp(loss, min=0.0)  # needed?
        return loss

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_input = super(EntmaxBisectLossFunction, cls).backward(ctx, grad_output)
        return grad_input + (None, None)


sparsemax_loss = SparsemaxLossFunction.apply
sparsemax_bisect_loss = SparsemaxBisectLossFunction.apply
entmax15_loss = Entmax15LossFunction.apply
entmax_bisect_loss = EntmaxBisectLossFunction.apply


class SparsemaxBisectLoss(_GenericLoss):

    def __init__(self, n_iter=50, weight=None, ignore_index=-100,
                 reduction='elementwise_mean'):
        self.n_iter = n_iter
        super(SparsemaxBisectLoss, self).__init__(weight, ignore_index, reduction)

    def loss(self, input, target):
        return sparsemax_bisect_loss(input, target, self.n_iter)


class SparsemaxLoss(_GenericLoss):

    def __init__(self, k=None, weight=None, ignore_index=-100,
                 reduction='elementwise_mean'):
        self.k = k
        super(SparsemaxLoss, self).__init__(weight, ignore_index, reduction)

    def loss(self, input, target):
        return sparsemax_loss(input, target, self.k)


class EntmaxBisectLoss(_GenericLoss):

    def __init__(self, alpha=1.5, n_iter=50, weight=None, ignore_index=-100,
                 reduction='elementwise_mean'):
        self.alpha = alpha
        self.n_iter = n_iter
        super(EntmaxBisectLoss, self).__init__(weight, ignore_index, reduction)

    def loss(self, input, target):
        return entmax_bisect_loss(input, target, self.alpha, self.n_iter)


class Entmax15Loss(_GenericLoss):

    def __init__(self, k=100, weight=None, ignore_index=-100,
                 reduction='elementwise_mean'):
        self.k = k
        super(Entmax15Loss, self).__init__(weight, ignore_index, reduction)

    def loss(self, input, target):
        return entmax15_loss(input, target, self.k)
