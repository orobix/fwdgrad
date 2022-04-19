from typing import Callable, Tuple
import torch
from torch.nn import functional as F

from fwdgrad.activation import softmax


def _xent(x: torch.Tensor, t: torch.Tensor, num_classes: int = 10):
    y = softmax(x)
    logy = -torch.log(y)
    loss = torch.mean(torch.sum(logy * F.one_hot(t, num_classes), dim=1))
    return loss


def xent(model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor, num_classes: int = 10):
    y = model(x)
    return _xent(y, t, num_classes)


def functional_xent(
    params: Tuple[torch.nn.Parameter],
    model: Callable[[Tuple[torch.nn.Parameter], torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    num_classes: int = 10,
):
    y = model(params, x)
    return _xent(y, t, num_classes)
