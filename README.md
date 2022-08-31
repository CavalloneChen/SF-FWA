# SF-FWA: A Self-Adaptive Fast Fireworks Algorithm for effective large-scale optimization
This repo holds the source code for our work published on Swarm adn Evolutionary Computation [[download the paper]](https://www.sciencedirect.com/science/article/pii/S2210650223000871) :
```
@article{chen2023sf,
  title={SF-FWA: A Self-Adaptive Fast Fireworks Algorithm for effective large-scale optimization},
  author={Chen, Maiyue and Tan, Ying},
  journal={Swarm and Evolutionary Computation},
  volume={80},
  pages={101314},
  year={2023},
  publisher={Elsevier}
}
```

## Installation
Clone the repo and create a conda environment with the `environment.yaml` file:
```bash
conda env create -f environment.yaml
```

## Run experiments for basic functions

```bash
python run_sffwa.py --mode basic
```

## Run experiments for CEC2013 LSGO
```bash
python run_sffwa.py --mode bench
```

## Run experiments for reinforcement learning
```bsah
python run_sffwa.py --mode mujoco
```


