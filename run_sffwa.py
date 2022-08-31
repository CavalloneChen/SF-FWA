import argparse
import os
import torch
from sffwa.sffwa import SFFWA
from sffwa.cec13lsgo_funcs_torch import CEC13LSGOFunction
from sffwa.basic_funs_torch import get_func
from sffwa.mujoco_funcs_torch import MujocoFunc
max_threads = 1

os.environ["OMP_NUM_THREADS"] = f"{max_threads}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{max_threads}"
os.environ["MKL_NUM_THREADS"] = f"{max_threads}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{max_threads}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{max_threads}"


torch.set_num_threads(max_threads)
torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="basic")
args = parser.parse_args()

if args.mode == "basic":
    max_evals = int(1e6)
    n_dim = int(1e3)
    device = torch.device("cuda")
    # device = torch.device("cpu")
    # func = get_func("NonRotated", "BentCigar", n_dim, device)
    # func = get_func("Rotated", "Discus", n_dim, device)
    func = get_func("Rotated", "BentCigar", n_dim, device)
    # func = get_func("Rotated", 'Ellipsoid', n_dim, device)
    # func = get_func("Rotated", "Rosenbrock", n_dim, device)
    # func = get_func("Mixed", "Ellipsoid", n_dim, device)
    # func = get_func("Mixed", 'BentCigar', n_dim, device)
    # func = get_func("Mixed", 'Discus', n_dim, device)
    # func = get_func("Rotated", "Schwefel", n_dim, device)
    # func = get_func("Rotated", "Schwefel", n_dim, device)

    common_hps = {
        "device": device,
        "evaluator": func,
        "lb": -5.0,
        "ub": 5.0,
        "n_dim": n_dim,
    }
    algo = SFFWA(rep_id=0, **common_hps)
    algo.run_eval(
        max_eval=max_evals,
        n_dumps=500,
        progress_bar=False,
        print_keys=[
            "evaluation",
            "best_sofar",
            "amp",
            "cond",
            "psi_mean",
            "zeta_mean",
            # "internal_runtime",
            # "eval_runtime",
        ],
    )
    
elif args.mode == "bench":
        max_evals = int(3e6)
        n_dim = 1000
        device = torch.device("cpu")
        func_id = 1
        func = CEC13LSGOFunction(func_id)
        common_hps = {
            "device": device,
            "evaluator": func,
            "lb": func.lb,
            "ub": func.ub,
            "n_dim": n_dim,
        }
        test_fwa = SFFWA(rep_id=0, **common_hps)
        test_fwa.run_eval(
            max_eval=max_evals,
            n_dumps=500,
            progress_bar=False,
            print_keys=[
                "evaluation",
                "best_sofar",
                "amp",
                "cond",
                "psi_mean",
                "zeta_mean",
            ],
        )
        
elif args.mode == "mujoco":
        max_evals = int(1e4)
        env_name = "HalfCheetah-v4"
        # env_name = "Walker2d-v4"
        # env_name = "Ant-v4"
        # env_name = "Hopper-v4"
        # env_name = "Swimmer-v4"
        func = MujocoFunc(
            env_name,
            hidden_layers=[128, 128],
        )
        n_dim = func.n_dim
        # device = torch.device("cpu")
        device = torch.device("cuda")
        common_hps = {
            "device": device,
            "evaluator": func,
            "lb": -100,
            "ub": 100,
            "n_dim": n_dim,
        }
        spec_hps = {
            "zero_init": True,
            "amp_init": 1e-1,
            "amp_lb": 1e-5,
            "amp_ub": 1.0,
            "psi_std": 0.6,
            "zeta_std": 0.8,
            "meta_lr": 1e-1,
        }
        test_fwa = SFFWA(rep_id=0, **common_hps, **spec_hps)
        print(f"Env: {env_name}, Ndim: {n_dim}, Lam: {test_fwa.lam}")
        test_fwa.run_eval(
            max_eval=max_evals,
            n_dumps=-1,
            progress_bar=False,
            stop_precision=-1e8,
            print_keys=[
                "evaluation",
                "n_gen",
                "best_reward",
                "pop_best_reward",
                "amp",
                "cond",
                "psi_mean",
                "zeta_mean",
                "grad_norm",
            ],
        )
else:
    raise Exception("Unknown mode, must choose fron [basic, bench, mujoco]")