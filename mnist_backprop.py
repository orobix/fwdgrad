import math
import time

import torch
import torchvision

import hydra
from omegaconf import DictConfig

from fwdgrad.loss import xent


@hydra.main(config_path="./configs/", config_name="config.yaml")
def train_model(cfg: DictConfig):
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device(f"cuda:{cfg.device_id}" if USE_CUDA else "cpu")
    if "NeuralNet" in cfg.model._target_:
        mnist = torchvision.datasets.MNIST(
            "/tmp/data",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Lambda(lambda x: torch.flatten(x)),
                ]
            ),
        )
        input_size = mnist.data.shape[1] * mnist.data.shape[2]
    else:
        mnist = torchvision.datasets.MNIST(
            "/tmp/data",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
        )
        input_size = 1  # Channel size

    train_loader = hydra.utils.instantiate(cfg.dataset, dataset=mnist)
    output_size = len(mnist.classes)

    model = hydra.utils.instantiate(cfg.model, input_size=input_size, output_size=output_size)
    model.to(DEVICE)
    model.float()
    model.train()
    params = tuple(model.parameters())

    t_total = 0
    for epoch in range(cfg.optimization.epochs):
        t0 = time.perf_counter()
        for i, batch in enumerate(train_loader):
            images, labels = batch
            loss = xent(model, images.to(DEVICE), labels.to(DEVICE))
            loss.backward()
            lr = lr = cfg.optimization.learning_rate * math.e ** (-(epoch * len(train_loader) + i) * cfg.optimization.k)
            for p in params:
                p.data.sub_(lr * p.grad.data)
                p.grad.data.zero_()
        t1 = time.perf_counter()
        t_total += t1 - t0
        print(f"Epoch [{epoch+1}/{cfg.optimization.epochs}], Loss: {loss.item():.4f}, Time (s): {t1 - t0:.4f}")
    print("Mean time:", t_total / cfg.optimization.epochs)


if __name__ == "__main__":
    train_model()
