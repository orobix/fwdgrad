import time

import torch
import torchvision
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig

from fwdgrad.loss import xent
from fwdgrad.model import NeuralNet


@hydra.main(config_path="./configs/", config_name="config.yaml")
def train_model(cfg: DictConfig):
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device(f"cuda:{cfg.device_id}" if USE_CUDA else "cpu")
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
    train_loader = DataLoader(
        mnist,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    input_size = mnist.data.shape[1] * mnist.data.shape[2]
    model = NeuralNet(input_size, cfg.model.hidden_dims)
    model.to(DEVICE)
    model.train()
    params = tuple(model.parameters())

    t_total = 0
    for epoch in range(cfg.optimization.epochs):
        t0 = time.perf_counter()
        for i, batch in enumerate(train_loader):
            images, labels = batch
            loss = xent(model, images.to(DEVICE), labels.to(DEVICE))
            loss.backward()
            for p in params:
                p.data.sub_(cfg.optimization.learning_rate * p.grad.data)
                p.grad.data.zero_()
        t1 = time.perf_counter()
        t_total += t1 - t0
        print(
            f"Epoch [{epoch+1}/{cfg.optimization.epochs}], Loss: {loss.item():.4f}, Time (s): {t1 - t0:.4f}"
        )
    print("Mean time:", t_total / cfg.optimization.epochs)


if __name__ == "__main__":
    train_model()
