import time
from functools import partial

import functorch as fc
import torch
import torchvision
from torch.utils.data import DataLoader

from fwdgrad.loss import functional_xent
from fwdgrad.model import NeuralNet

EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
HIDDEN_SIZES = [64]


def train_model():
    mnist = torchvision.datasets.MNIST(
        "/tmp/data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambda x: torch.flatten(x))]
        ),
    )
    train_loader = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    with torch.no_grad():
        model = NeuralNet(784, HIDDEN_SIZES)
        model.train()

        # Get the functional version of the model with functorch
        fmodel, params = fc.make_functional(model)
        params = tuple(params)

        t_total = 0
        for epoch in range(EPOCHS):
            t0 = time.perf_counter()
            for i, batch in enumerate(train_loader):
                images, labels = batch

                # Sample tangent vectors for every parameters of the model
                v_params = tuple([torch.randn_like(p) for p in params])
                f = partial(functional_xent, model=fmodel, x=images, t=labels)
                loss, jvp = fc.jvp(f, (params,), (v_params,))
                params = tuple([p.sub_(LR * jvp * v_params[i]) for i, p in enumerate(params)])
            t1 = time.perf_counter()
            t_total += t1 - t0
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}, Time (s): {t1 - t0:.4f}")
        print(f"Mean time: {t_total / EPOCHS:.4f}")


if __name__ == "__main__":
    train_model()
