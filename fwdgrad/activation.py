import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax function.

    Args:
        x (torch.Tensor): tensor over which to apply softmax.
        dim (int, optional): dimension over which to apply softmax. Defaults to 1.

    Returns:
        torch.Tensor: softmax of x.
    """
    maxes = torch.max(x, dim, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
    return torch.div(x_exp, x_exp_sum)
