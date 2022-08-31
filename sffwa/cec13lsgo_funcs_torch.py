import numpy as np
import torch
from cec2013lsgo.cec2013 import Benchmark as CEC13LSGO

bench = CEC13LSGO()


def get_cec13lsgo_func(func_id):
    func = bench.get_function(func_id)
    lb = bench.get_info(func_id)["lower"]
    ub = bench.get_info(func_id)["upper"]
    return {
        "func": func,
        "lb": lb,
        "ub": ub,
    }


class CEC13LSGOFunction:
    def __init__(self, func_id: int) -> None:
        self.func_id = func_id
        f_dict = get_cec13lsgo_func(func_id)
        self.f = f_dict["func"]
        self.lb = f_dict["lb"]
        self.ub = f_dict["ub"]

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        assert X.ndim == 2
        device = X.device
        return torch.tensor([self.f(x.cpu().numpy()) for x in X], device=device)
