import math


def exponential_lr_decay(step: int, k: float):
    return math.e ** (-step * k)
