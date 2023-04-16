import os
import numpy as np


def TotalGradNorm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item()**norm_type
    total_norm = total_norm**(1. / norm_type)
    return total_norm


def log_train(iter_i, loss, psnr, grad_norm):
    with open(f'{os.environ["LOG_DIR"]}/loss.txt', 'a') as f:
        f.write(f'{iter_i},{loss},{grad_norm},{psnr}\n')
    f.close()


def log_cfg(func):
    def wrap(*args, **kwargs):
        with open(f'{os.environ["LOG_DIR"]}/cfg.txt', 'a') as f:
            f.write('='*100+'\n')
            f.write(f"{func.__name__}:\n")
            f.write(f"\tKwargs\n")
            for k, v in kwargs.items():
                msg = v
                if isinstance(v, list):
                    msg = f"{type(v)}:{len(v)}"
                if isinstance(v, np.ndarray):
                    msg = f"{type(v)}:{v.shape}"

                f.write(f"\t\t{k}:{msg}\n")

            f.write(f"\targs\n")
            for v in args:
                msg = v
                if isinstance(v, list):
                    msg = f"{type}:{len(v)}"
                f.write(f"\t\t{msg}\n")
        f.close()

        res = func(*args, **kwargs)
        return res
    wrap.__name__ = func.__name__
    return wrap
