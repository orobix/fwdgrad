import torch

def clamp_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)