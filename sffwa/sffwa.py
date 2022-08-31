import argparse
import math
import os
import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sffwa.basic_funs_torch import get_func
from sffwa.cec13lsgo_funcs_torch import CEC13LSGOFunction
from sffwa.mujoco_funcs_torch import MujocoFunc

class BaseOptimizer(ABC):
    @abstractmethod
    def step_gen(self):
        pass

    def run_eval(
        self,
        max_eval: int,
        print_keys: list = [],
        n_dumps: int = 100,
        dump_path: str = None,
        progress_bar: bool = True,
        stop_precision: float = 1e-10,
        record_runtime: bool = False,
    ) -> pd.DataFrame:
        self.dump_path = dump_path
        self.stop_precision = stop_precision
        self.max_eval = max_eval
        self.log_dict = {"elapsed_time": []}
        self.start_time = time.time()
        self.record_runtime = record_runtime
        self.log_milestone_list = np.arange(0, max_eval, 1e3).tolist()
        self.n_dumps = n_dumps
        self.dump_milestone_list = np.arange(0, max_eval, max_eval / n_dumps).tolist()
        self.best_sofar = float("inf")
        self.eval_marker = 0
        self.df = None
        if progress_bar:
            pbar = tqdm(total=max_eval)
        # with tqdm(total=max_eval) as pbar:
        while True:
            res = self.step_gen()
            if self.should_log():
                self.update_df(res)
                if self.log_milestone_list:
                    self.log_milestone_list.pop(0)
                latest_evals = res["evaluation"]
                accum_evals = latest_evals - self.eval_marker
                if progress_bar:
                    pbar.update(accum_evals)
                self.eval_marker = latest_evals

            if self.should_dump():
                if print_keys:
                    row = " | ".join(
                        [f"{k}:{self.log_dict[k][-1]:.2E}" for k in print_keys]
                    )
                    row = f"| {row} |"
                    print(row)
                if self.dump_path:
                    self.df.to_csv(self.dump_path)
                if self.dump_milestone_list:
                    self.dump_milestone_list.pop(0)

            if self.should_stop():
                self.update_df(res)
                if self.dump_path:
                    self.df.to_csv(self.dump_path)
                break
        if progress_bar:
            pbar.close()

        return self.df

    def remap(self, X):
        l_mask = X < self.lb
        u_mask = X > self.ub
        X = torch.where(l_mask, self.lb, X)
        X = torch.where(u_mask, self.ub, X)
        return X

    def update_df(self, res: dict):
        self.log_dict["elapsed_time"].append(time.time() - self.start_time)
        for k, v in res.items():
            if k not in self.log_dict:
                self.log_dict[k] = []
            self.log_dict[k].append(v)
        self.df = pd.DataFrame(self.log_dict)
        self.df["rep_id"] = self.rep_id
        # eval_milestones = self.log_dict["evaluation"]
        self.best_sofar = self.log_dict["best_sofar"][-1]

    def should_stop(self):
        return (self.n_evals > self.max_eval) or (self.best_sofar < self.stop_precision)

    def should_log(self):
        if self.n_dumps == -1:
            return True
        else:
            return (len(self.log_milestone_list) == 0) or (
                self.n_evals >= self.log_milestone_list[0]
            )

    def should_dump(self):
        if self.n_dumps == -1:
            return True
        else:
            return (len(self.dump_milestone_list) == 0) or (
                self.n_evals >= self.dump_milestone_list[0]
            )


class SFFWA(BaseOptimizer):
    def __init__(
        self,
        rep_id: int,
        device,
        evaluator,
        lb: float,
        ub: float,
        n_dim: int,
        n_fireworks: int = 4,
        zero_init: bool = False,
        amp_init=None,
        psi_std: float = 0.1,
        zeta_std: float = 0.3,
        meta_lr: float = 1e-2,
        c1: float = 0.93,
        m: int = 10,
        cs: float = 0.3,
        q_star: float = 0.7,
        cc: float = None,
        cvar: float = None,
        amp_lb: float = 1e-12,
        amp_ub: float = 1e8,
        momentum: float = 0.0,
        fixed_psi: float = None,
        fixed_zeta: float = None,
    ):
        self.rep_id = rep_id
        self.device = device
        self.eval = evaluator
        self.lb = lb
        self.ub = ub
        self.n_dim = n_dim
        self.fixed_psi = fixed_psi
        self.fixed_zeta = fixed_zeta
        self.n_fireworks = n_fireworks
        self.n_sparks = (4 + int(math.floor(3 * math.log(self.n_dim)))) // 2
        if self.n_sparks % 2 != 0:
            self.n_sparks += 1
        self.lam = int(self.n_sparks * self.n_fireworks)
        self.c1 = c1

        self.m = m
        self.q_gen_gap = self.n_dim

        self.amp_lb = amp_lb
        self.amp_ub = amp_ub
        self.beta = momentum

        self.mu_ratio = 0.5
        self.mu_sparks = int(self.n_sparks * self.mu_ratio)
        self.mu_global = int(int(self.n_fireworks * self.n_sparks) * self.mu_ratio)
        self.ws = self.get_recomb_ws(self.mu_sparks)
        self.gws = self.get_recomb_ws(self.mu_global)
        self.global_mueff = (1 / torch.sum(self.gws**2)).item()
        self.local_mueff = (1 / torch.sum(self.ws**2)).item()
        self.cs = cs
        self.q_star = q_star
        if cc is not None:
            self.cc = cc
        else:
            self.cc = 10 / (self.n_dim + 31)
        if cvar is not None:
            self.cvar = cvar
        else:
            self.cvar = 10 / (self.n_dim + 31)
        self.mirror_sampling = True

        self.psi_mean = torch.tensor([-2.0], device=self.device)
        self.zeta_mean = torch.tensor([0.0], device=self.device)
        if psi_std is not None:
            self.psi_std = psi_std
        else:
            self.psi_std = 0.1
        if zeta_std is not None:
            self.zeta_std = zeta_std
        else:
            self.zeta_std = 0.3
        if meta_lr is not None:
            self.meta_lr = meta_lr
        else:
            self.meta_lr = 1e-2

        self.hist_best_y = np.inf
        self.n_evals = 0
        self.n_gen = 0
        self.n_restarts = 0

        if zero_init:
            self.firework_x = torch.zeros(size=[1, self.n_dim], device=self.device)
        else:
            self.firework_x = (
                (torch.rand(size=[1, self.n_dim], device=self.device) - 0.5)
                + (self.ub + self.lb) / 2
            ) * (self.ub - self.lb)
        self.cur_best_y = np.inf
        if amp_init is not None:
            self.amp = amp_init
        else:
            self.amp = (self.ub - self.lb) * 0.3

        self.s = 0
        self.prevbests = torch.ones(self.mu_global, device=self.device) * np.inf
        self.PCs = torch.zeros(size=[self.m, self.n_dim], device=self.device)
        self.pcidxs = torch.arange(self.m, device=self.device)
        self.itrs = torch.arange(self.m, device=self.device)
        self.global_pc = torch.zeros(self.n_dim, device=self.device)
        self.global_var = torch.ones(self.n_dim, device=self.device)
        self.velocity = torch.zeros(self.n_dim, device=self.device)
        self.adam_m = torch.zeros(self.n_dim, device=self.device)
        self.adam_v = torch.zeros(self.n_dim, device=self.device)

        if self.mirror_sampling:
            n_samples = self.n_sparks // 2
        else:
            n_samples = self.n_sparks

        self.Z1 = torch.randn(
            size=[self.n_fireworks, n_samples, self.n_dim],
            device=self.device,
        )
        self.Z2 = torch.randn(
            size=[self.n_fireworks, n_samples, self.n_dim],
            device=self.device,
        )
        self.ZV = torch.randn(
            size=[self.n_fireworks, n_samples, self.m],
            device=self.device,
        )

        self.Zhp = torch.randn(
            size=[self.n_fireworks // 2],
            device=self.device,
        )

    def sample_target(
        self,
        mean: float,
        std: float,
    ):
        self.Zhp.normal_()
        sample_1 = self.Zhp * std
        sample_2 = -sample_1
        samples = torch.cat([sample_1, sample_2]) + mean
        return samples

    def get_recomb_ws(self, mu: int) -> torch.Tensor:
        ln_mu_sum = np.sum([math.log(i) for i in range(1, mu + 1)])
        return torch.tensor(
            [
                (math.log(mu + 1) - math.log(i)) / (mu * math.log((mu + 1)) - ln_mu_sum)
                for i in range(1, mu + 1)
            ],
            device=self.device,
        )

    def get_sgdm_step(self, grad_est):
        beta = 0.9
        lr = 1e-2
        self.velocity = beta * self.velocity + (1 - beta) * grad_est
        return lr * self.velocity

    def get_adam_step(self, grad_est):
        beta1 = 0.99
        beta2 = 0.999
        lr = 1e-2
        self.adam_m = (1 - beta1) * self.adam_m + beta1 * self.adam_m
        self.adam_v = (1 - beta2) * (grad_est**2) + beta2 * self.adam_v
        mhat = self.adam_m / (1 - (beta1**self.n_gen))
        vhat = self.adam_v / (1 - (beta2**self.n_gen))
        return lr * mhat / (torch.sqrt(vhat) + 1e-8)

    def step_gen(self):
        gen_start_time = time.time()

        # * Sampling
        sampled_psis = self.sample_target(
            self.psi_mean,
            self.psi_std,
        ).clamp(-2, -1)
        sampled_zetas = self.sample_target(
            self.zeta_mean,
            self.zeta_std,
        ).clamp(-5, 5)
        self.Z1.normal_()  # [self.n_fireworks, self.n_sparks//2, self.n_dim]
        self.Z2.normal_()  # [self.n_fireworks, self.n_sparks//2, self.n_dim]
        self.ZV.normal_()  # [self.n_fireworks, self.n_sparks//2, self.m]

        if self.fixed_psi is None and self.fixed_zeta is None:
            psis = torch.pow(10, sampled_psis)
            zetas = 1 / (1 + torch.pow(10, -sampled_zetas))
        else:
            psis = torch.tensor([self.fixed_psi for _ in range(self.n_fireworks)])
            zetas = torch.tensor([self.fixed_zeta for _ in range(self.n_fireworks)])
        a = torch.sqrt(1 - psis).view(-1, 1, 1)
        b = torch.sqrt(psis).view(-1, 1, 1)
        # c = (1 - zetas).view(-1, 1, 1)
        # d = torch.sqrt(zetas * (2 - zetas)).view(-1, 1, 1)
        c = torch.sqrt(1 - zetas).view(-1, 1, 1)
        d = torch.sqrt(zetas).view(-1, 1, 1)
        X1 = self.amp * (
            a * ((c * torch.sqrt(self.global_var)) * self.Z1 + d * self.Z2)
            + b * torch.matmul(self.ZV, self.PCs) / math.sqrt(self.m)
        )
        if self.mirror_sampling:
            X2 = -X1
            X = torch.cat([X1, X2], dim=1) + self.firework_x
        else:
            X = X1 + self.firework_x
        X = X.view(-1, self.n_dim)
        X = self.remap(X)
        eval_start_time = time.time()
        Y = self.eval(X)
        eval_end_time = time.time()

        sorted_idxs = torch.argsort(Y)
        sorted_Y = Y[sorted_idxs]
        pop_best_idx = sorted_idxs[0]
        pop_best_x = X[pop_best_idx]
        pop_best_y = Y[pop_best_idx]
        gmu_idxs = sorted_idxs[: self.mu_global]
        global_elite_X = X[gmu_idxs]

        global_delta_mean = self.gws @ global_elite_X - self.firework_x

        self.global_pc = (1 - self.cc) * self.global_pc + math.sqrt(
            self.cc * (2 - self.cc) * self.global_mueff
        ) * (global_delta_mean / self.amp)
        self.global_var = (1 - self.cvar) * self.global_var + self.cvar * (
            (self.gws @ (((global_elite_X - self.firework_x) / self.amp) ** 2))
        )

        grad_step = global_delta_mean
        self.firework_x = self.firework_x + grad_step

        # * Step-size adaptation
        cur_bests = sorted_Y[: self.mu_global]
        new_union_bests = torch.sort(
            torch.cat([cur_bests, self.prevbests], dim=0), descending=False
        ).values[: self.mu_global]

        Q = torch.sum(
            self.gws[new_union_bests < self.prevbests[: self.mu_global]]
        ).item()
        self.s = (1 - self.cs) * self.s + self.cs * (Q - self.q_star)
        if self.s < 0:
            self.amp *= self.c1
        else:
            self.amp *= 1 / self.c1
        self.prevbests = new_union_bests
        self.amp = np.clip(self.amp, self.amp_lb, self.amp_ub)

        # * Evolution path archive udpate
        if self.n_gen <= self.m:
            self.PCs[self.n_gen - 1] = self.global_pc
        else:
            valmin, imin = torch.min(
                self.itrs[self.pcidxs[1:]] - self.itrs[self.pcidxs[:-1]], dim=0
            )
            valmin, imin = valmin.item(), imin.item()
            imin = imin + 1
            if valmin > self.q_gen_gap:
                imin = 0
            self.pcidxs = self.pcidxs[
                list(range(imin)) + list(range(imin + 1, self.m, 1)) + [imin]
            ]
            self.itrs[self.pcidxs[self.m - 1]] = self.n_gen
            self.PCs[self.pcidxs[self.m - 1]] = self.global_pc
        self.cur_best_y = min(self.cur_best_y, Y[pop_best_idx].item())
        self.n_evals += len(Y)
        self.n_gen += 1

        # * Hyper-parameter adaptation
        aY = Y.view(self.n_fireworks, self.n_sparks)
        aY = torch.sort(aY, dim=-1, descending=False).values[:, : self.mu_sparks]
        sbc = torch.argmin(aY, dim=0)
        self.psi_mean = (1 - self.meta_lr) * self.psi_mean + self.meta_lr * (
            self.ws @ sampled_psis[sbc]
        )
        self.zeta_mean = (1 - self.meta_lr) * self.zeta_mean + self.meta_lr * (
            self.ws @ sampled_zetas[sbc]
        )

        gen_end_time = time.time()
        gen_eval_time = eval_end_time - eval_start_time
        gen_internal_time = (gen_end_time - gen_start_time) - gen_eval_time

        res = {
            "evaluation": self.n_evals,
            "n_gen": self.n_gen,
            "n_restarts": self.n_restarts,
            "best_sofar": self.cur_best_y,
            "pop_best": pop_best_y.item(),
            "best_reward": -self.cur_best_y,
            "pop_best_reward": -pop_best_y.item(),
            "amp": self.amp,
            "psi_mean": self.psi_mean.item(),
            "zeta_mean": self.zeta_mean.item(),
            "cond": torch.sqrt(
                torch.max(self.global_var) / torch.min(self.global_var)
            ).item(),
            "internal_runtime": gen_internal_time,
            "eval_runtime": gen_eval_time,
            "grad_norm": torch.linalg.norm(grad_step).item() / math.sqrt(self.n_dim),
        }
        return res





def rank_tensor_elements(x: torch.Tensor, descending: bool = True):
    sorted_idx = x.argsort(descending=descending)
    ranks = torch.empty_like(sorted_idx)
    ranks[sorted_idx] = torch.arange(len(x), device=x.device)
    return ranks
