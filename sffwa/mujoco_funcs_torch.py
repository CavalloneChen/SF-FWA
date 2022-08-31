import copy

import gym
import torch
from torch import nn
from tqdm import tqdm


def net_to_vec(network_ins: nn.Module) -> torch.Tensor:
    # * Convert a network's parameters to a single vector
    return nn.utils.parameters_to_vector(network_ins.parameters())


def net_from_vec(net_template: nn.Module, param_vec: torch.Tensor) -> nn.Module:
    # * Replace a network's parameters from a single vector
    out_net = copy.deepcopy(net_template)
    nn.utils.vector_to_parameters(param_vec, out_net.parameters())
    return out_net


class Mlp(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list,
        act_bound: float=1.0,
        mid_act=nn.Tanh,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.mid_act = mid_act
        self.layer_norm = layer_norm
        self.act_bound = act_bound
        self.layers = self.make_layers()

    def make_layers(self):
        # input_layer = [nn.Linear(self.input_dim, self.hidden_dims[0]), nn.Tanh()]
        input_layer = []
        input_layer.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        if self.layer_norm:
            input_layer.append(nn.LayerNorm(self.hidden_dims[0]))
        input_layer.append(self.mid_act())

        hidden_layers = []
        for i in range(len(self.hidden_dims) - 1):
            h1, h2 = self.hidden_dims[i], self.hidden_dims[i + 1]
            hidden_layers.append(nn.Linear(h1, h2))
            if self.layer_norm:
                hidden_layers.append(nn.LayerNorm(h2))
            hidden_layers.append(self.mid_act())

        output_layer = [nn.Linear(self.hidden_dims[-1], self.output_dim), nn.Tanh()]

        layers = input_layer + hidden_layers + output_layer
        layers = nn.Sequential(*layers)
        return layers

    def forward(self, x):
        return self.layers(x) * self.act_bound


class MujocoFunc:
    def __init__(self, env_name: str, hidden_layers: list=[32,32]):
        self.env_name = env_name
        tmp_env = gym.make(self.env_name)
        self.env_obs_dim = tmp_env.observation_space.shape[0]
        self.env_act_dim = tmp_env.action_space.shape[0]
        if self.env_name.startswith("Humanoid"):
            act_bound = 0.4
        else:
            act_bound = 1.0
        self.net = Mlp(self.env_obs_dim, self.env_act_dim, hidden_layers, act_bound=act_bound)
        self.n_dim = len(net_to_vec(self.net))
        self.max_steps_per_episode = 1000

    def rollout(
        self,
        env,
        actor,
        device,
    ):
        done = False
        ep_rew = 0
        rolled_steps = 0
        cur_obs, _ = env.reset()
        while not done:
            action = (
                actor(torch.Tensor(cur_obs.copy()).to(device)).detach().cpu().numpy()
            )
            next_obs, reward, done, *_ = env.step(action)
            rolled_steps += 1
            ep_rew += reward
            if rolled_steps == self.max_steps_per_episode:
                break
            cur_obs = next_obs
        env.close()
        return {
            "rolled_steps": rolled_steps,
            "ep_reward": ep_rew,
        }

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        assert X.ndim == 2
        device = X.device
        Y = []
        for vec in tqdm(X):
            net = net_from_vec(self.net, vec)
            res = self.rollout(gym.make(self.env_name), net, device)
            ep_rew = res["ep_reward"]
            Y.append(ep_rew)
        return -torch.tensor(Y, device=device)
    
if __name__ == "__main__":
    env_name = "HalfCheetah-v4"
    f = MujocoFunc(env_name)
    n_dim = f.n_dim
    X = torch.randn((5, n_dim)) * 1e-3
    Y = f(X)
    print(Y)
    