import torch


def baele(x):
    return (
        (torch.tensor([1.5]) - x[0] + x[0] * x[1]) ** 2
        + (torch.tensor([2.25]) - x[0] + x[0] * x[1] ** 2) ** 2
        + (torch.tensor([2.625]) - x[0] + x[0] * x[1] ** 3) ** 2
    )


def rosenbrock(x):
    return (torch.tensor([1]) - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
