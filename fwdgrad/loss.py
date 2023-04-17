from typing import Dict, KeysView, ValuesView

import torch
import torch.func as fc
from torch import Tensor
from torch.nn import functional as F


def _xent(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss.

    Args:
        x (torch.Tensor): Output of the model.
        t (torch.Tensor): Targets.

    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    return F.cross_entropy(x, t)


def xent(model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss. Given a pytorch model, it computes the cross-entropy loss.

    Args:
        model (torch.nn.Module): PyTorch model.
        x (torch.Tensor): Input tensor for the PyTorch model.
        t (torch.Tensor): Targets.

    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    y = model(x)
    return _xent(y, t)


def functional_xent(
    params: ValuesView,
    buffers: Dict[str, Tensor],
    names: KeysView,
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Functional cross-entropy loss. Given a pytorch model it computes the cross-entropy loss
    in a functional way.

    Args:
        params: Model parameters.
        buffers: Buffers of the model.
        names: Names of the parameters.
        model: A pytorch model.
        x (torch.Tensor): Input tensor for the PyTorch model.
        t (torch.Tensor): Targets.

    Returns:
        torch.Tensor: Cross-entropy loss.
    """
    y = fc.functional_call(model, ({k: v for k, v in zip(names, params)}, buffers), (x,))
    return _xent(y, t)
