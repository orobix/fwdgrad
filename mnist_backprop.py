import time

import torch
import torchvision
from torch.utils.data import DataLoader

from fwdgrad.loss import xent
from fwdgrad.model import NeuralNet

EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
HIDDEN_SIZES = [64]
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


def train_model():
    mnist = torchvision.datasets.MNIST(
        "/tmp/data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambda x: torch.flatten(x))]
        ),
    )
    train_loader = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    model = NeuralNet(784, HIDDEN_SIZES)
    model.to(DEVICE)
    model.train()
    params = tuple(model.parameters())

    t_total = 0
    for epoch in range(EPOCHS):
        t0 = time.perf_counter()
        for i, batch in enumerate(train_loader):
            images, labels = batch
            loss = xent(model, images.to(DEVICE), labels.to(DEVICE))
            loss.backward()
            for p in params:
                p.data.sub_(LR * p.grad.data)
                p.grad.data.zero_()
        t1 = time.perf_counter()
        t_total += t1 - t0
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}, Time (s): {t1 - t0:.4f}")
    print("Mean time:", t_total / EPOCHS)


if __name__ == "__main__":
    train_model()
