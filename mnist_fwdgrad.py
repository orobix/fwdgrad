import math
import time
from functools import partial

import functorch as fc
import hydra
import torch
import torchvision
from functorch import make_functional
from omegaconf import DictConfig

from fwdgrad.loss import functional_xent


@hydra.main(config_path="./configs/", config_name="config.yaml")
def train_model(cfg: DictConfig):
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{cfg.device_id}" if use_cuda else "cpu")
    total_epochs = cfg.optimization.epochs
    init_lr = cfg.optimization.learning_rate
    k = cfg.optimization.k
    
    # Dataset creation
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
    with torch.no_grad():
        model = hydra.utils.instantiate(cfg.model, input_size=input_size, output_size=output_size)
        model.to(device)
        model.float()
        model.train()

        # Get the functional version of the model with functorch
        fmodel, params = make_functional(model)

        t_total = 0
        for epoch in range(total_epochs):
            t0 = time.perf_counter()
            for i, batch in enumerate(train_loader):
                images, labels = batch

                # Sample perturbation (tangent) vectors for every parameter of the model
                v_params = tuple([torch.randn_like(p) for p in params])
                f = partial(
                    functional_xent,
                    model=fmodel,
                    x=images.to(device),
                    t=labels.to(device),
                )

                # Forward AD
                loss, jvp = fc.jvp(f, (tuple(params),), (v_params,))

                # Forward gradient + parmeter update (SGD)
                lr = init_lr * math.e ** (-(epoch * len(train_loader) + i) * k)
                for j, p in enumerate(params):
                    p.sub_(lr * jvp * v_params[j])
            t1 = time.perf_counter()
            t_total += t1 - t0
            print(f"Epoch [{epoch+1}/{total_epochs}], Loss: {loss.item():.4f}, Time (s): {t1 - t0:.4f}")
        print(f"Mean time: {t_total / total_epochs:.4f}")


if __name__ == "__main__":
    train_model()
