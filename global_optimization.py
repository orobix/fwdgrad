import functorch as fc
import torch
import time


def baele(x):
    return (
        (torch.tensor([1.5]) - x[0] + x[0] * x[1]) ** 2
        + (torch.tensor([2.25]) - x[0] + x[0] * x[1] ** 2) ** 2
        + (torch.tensor([2.625]) - x[0] + x[0] * x[1] ** 3) ** 2
    )


def rosenbrock(x):
    return (torch.tensor([1]) - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


ITERATIONS = 1000
LEARNING_RATE = 1e-2


def main():
    params = torch.rand(2)
    t_total = 0
    for iteration in range(ITERATIONS):
        t0 = time.perf_counter()
        v_params = torch.randn_like(params)
        func_value, jvp = fc.jvp(baele, (params,), (v_params,))
        params = params.sub_(LEARNING_RATE * jvp * v_params)
        t1 = time.perf_counter()
        t_total += t1 - t0

        if iteration % 199 == 0:
            print(
                f"Iteration [{iteration + 1}/{ITERATIONS}], Loss: {func_value.item():.4f}, Time (s): {t1 - t0:.4f}"
            )

    print(f"Total time: {t_total:.4f}")
    print(f"Parameters value:\n" f"\tx: {params[0]}\n" f"\ty: {params[1]}")


if __name__ == "__main__":
    main()
