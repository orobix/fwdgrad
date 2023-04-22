import os
import time

import hydra
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import DictConfig, OmegaConf
from torch.utils import tensorboard

from fwdgrad.loss import xent

OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)


@hydra.main(config_path="./configs/", config_name="config.yaml")
def train_model(cfg: DictConfig):
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{cfg.device_id}" if use_cuda else "cpu")
    total_epochs = cfg.epochs
    grad_clipping = cfg.grad_clipping

    # Summary
    writer = tensorboard.writer.SummaryWriter(os.path.join(os.getcwd(), "logs/backprop"))

    # Dataset creation
    input_size = 1  # Channel size
    transform = [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
    if "NeuralNet" in cfg.model._target_:
        transform.append(torchvision.transforms.Lambda(torch.flatten))
        mnist_train = torchvision.datasets.MNIST(
            "/tmp/data",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(transform),
        )
        mnist_test = torchvision.datasets.MNIST(
            "/tmp/data",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(transform),
        )
        input_size = mnist_train.data.shape[1] * mnist_train.data.shape[2]
    else:
        mnist_train = torchvision.datasets.MNIST(
            "/tmp/data",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(transform),
        )
        mnist_test = torchvision.datasets.MNIST(
            "/tmp/data",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(transform),
        )
    train_loader = hydra.utils.instantiate(cfg.dataset, dataset=mnist_train)
    test_loader = hydra.utils.instantiate(cfg.dataset, dataset=mnist_test)

    output_size = len(mnist_train.classes)
    model: torch.nn.Module = hydra.utils.instantiate(cfg.model, input_size=input_size, output_size=output_size)
    model.to(device)
    model.float()
    model.train()
    params = model.parameters()

    optimizer: torch.optim.Optimizer = hydra.utils.instantiate(cfg.optimizer, params=params)
    optimizer.zero_grad(set_to_none=True)

    scheduler: torch.optim.lr_scheduler._LRScheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

    steps = 0
    t_total = 0.0
    for epoch in range(total_epochs):
        t0 = time.perf_counter()
        for batch in train_loader:
            steps += 1
            images, labels = batch
            loss = xent(model, images.to(device), labels.to(device))
            loss.backward()
            if grad_clipping > 0:
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    parameters=params, max_norm=grad_clipping, error_if_nonfinite=True
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            writer.add_scalar("Loss/train_loss", loss, steps)
            writer.add_scalar("Misc/lr", scheduler.get_last_lr()[0], steps)
        t1 = time.perf_counter()
        t_total += t1 - t0
        writer.add_scalar("Time/batch_time", t1 - t0, steps)
        writer.add_scalar("Time/sps", steps / t_total, steps)
        print(f"Epoch [{epoch+1}/{total_epochs}], Loss: {loss.item():.4f}, Time (s): {t1 - t0:.4f}")
    print("Mean time:", t_total / total_epochs)

    # Test
    acc = 0
    for batch in test_loader:
        images, labels = batch
        out = model(images.to(device))
        pred = F.softmax(out, dim=-1).argmax(dim=-1)
        acc += (pred == labels.to(device)).sum()
    writer.add_scalar("Test/accuracy", acc / len(mnist_test), steps)
    print(f"Test accuracy: {(acc / len(mnist_test)).item():.4f}")


if __name__ == "__main__":
    train_model()
