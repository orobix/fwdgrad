from typing import List, Optional

import torch
from torch import nn


class NeuralNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation_function: Optional[torch.nn.Module] = None,
    ):
        """Standard Fully-Connnected layers.

        Args:
            input_size (int): input size of the model.
            hidden_sizes (List[int]): a list of hidden sizes.
            output_size (int): The number of output classes.
            activation_function (Optional[torch.nn.Module], optional): the activation function for the hidden layers.
                Defaults to None.

        """
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.hidden_sizes.insert(0, input_size)
        for i in range(len(hidden_sizes) - 1):
            setattr(self, f"fc{i}", nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            setattr(self, f"act{i}", activation_function or nn.ReLU())
        self.out = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.hidden_sizes) - 1):
            x = getattr(self, f"act{i}")(getattr(self, f"fc{i}")(x))
        x = self.out(x)
        return x
