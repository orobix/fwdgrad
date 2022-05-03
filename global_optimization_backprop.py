import functorch as fc
import torch
import time
import hydra


@hydra.main(config_path="./configs/", config_name="global_optim_config.yaml")
def main(cfg):
    torch.manual_seed(cfg.seed)
    params = torch.rand(2)
    t_total = 0
    function = hydra.utils.instantiate(
        cfg.function,
    )

    for iteration in range(cfg.iterations):
        t0 = time.perf_counter()
        params.requires_grad_(True)
        func_value = function(params)
        func_value.backward()
        params = params.data.sub_(cfg.learning_rate * params.grad.data)
        t1 = time.perf_counter()
        t_total += t1 - t0

        if iteration % 199 == 0:
            print(
                f"Iteration [{iteration + 1}/{cfg.iterations}], Loss: {func_value.item():.4f}, Time (s): {t1 - t0:.4f}"
            )

    print(f"Total time: {t_total:.4f}")
    print(f"Parameters value:\n" f"\tx: {params[0]}\n" f"\ty: {params[1]}")


if __name__ == "__main__":
    main()
