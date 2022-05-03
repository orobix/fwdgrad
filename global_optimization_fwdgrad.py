import functorch as fc
import torch
import time
import hydra


@hydra.main(config_path="./configs/", config_name="global_optim_config.yaml")
def main(cfg):
    torch.manual_seed(cfg.seed)
    params = torch.rand(2)
    t_total = 0
    for iteration in range(cfg.iterations):
        t0 = time.perf_counter()

        # Sample perturbation vector
        v_params = torch.randn_like(params)

        # Forward AD
        func_value, jvp = fc.jvp(
            hydra.utils.call(
                cfg.function,
            ),
            (params,),
            (v_params,),
        )

        # Forward gradient + parmeter update (SGD)
        params = params.sub_(cfg.learning_rate * jvp * v_params)

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
