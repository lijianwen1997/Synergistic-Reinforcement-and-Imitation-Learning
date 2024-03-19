# Synergistic Reinforcement and Imitation Learning (SRIL) for Vision-driven Autonomous Flight of UAV Along River

Author: Zihan Wang, Jianwen Li and Nina Mahmoudian

Mechanical Engineering, Purdue University 

[![Video](https://img.youtube.com/vi/NJVux-3tXqA/hqdefault.jpg)](https://www.youtube.com/watch?v=NJVux-3tXqA)

## Installation
- The Riverine Simulation Environment has only been tested on Ubuntu 20.04 and Ubuntu 22.04.

- The Unity riverine environments can be downloaded from Google Drive [link](https://drive.google.com/file/d/16LXTkudfFbzxL5ZDCC1b4st59ybRoNpn/view?usp=sharing), then unzipped to the folder 'riverine_simulation'.

- The VAE model can be downloaded from this [link](https://drive.google.com/file/d/1SVU3p5wbGQnQs7U3qp7Gz0eCdo7YrYGh/view?usp=sharing), then put it to 'encoder/models/vae-sim-rgb-all.pth'.

- [Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) is encouraged to build the virtual python environment, and Python 3.9 is the supported version.

```shell
conda create -n SRIL python==3.9 && conda activate SRIL
```

- Install package dependencies
```shell
pip install -r requirements.txt
```

- If have error installing gym during the above process, install the specific version of setuptools then pip install again
```shell
pip install -U setuptools==65.5.0 
```

- Run the `riverine_simulation/collect_demonstrations.py` to control the UAV agent in the Unity riverine environment to test successful installation.
Keys **W** and **S** control camera altitude, **A** and **D** control yaw, **I** and **K** control longitudinal movement, **J** and **L** control latitudinal movement.

https://github.com/lijianwen1997/Synergistic-Reinforcement-and-Imitation-Learning/assets/28019327/1c6c9165-bc6c-42c4-84f5-f55b9730829f


- If have protobuf error running the Unity environment, downgrade the protobuf version
```shell
pip install -U protobuf==3.20
```

- If you want to play the CliffCircular environment, run `python -m cliff_circular.cliffcircular` or `python -m cliff_circular.cliffcircular_gym`.

https://github.com/lijianwen1997/Synergistic-Reinforcement-and-Imitation-Learning/assets/28019327/105283ff-e218-46b6-9771-c602787519da



## Getting started

- To train BC expert in the CliffCircular environment, run 
```
python -m sril.train_bc_cliff
```

- To train BC expert in the Unity riverine environment, run 
```
python -m sril.train_bc_riverine
```

- To train PPO+DynamicBC agent in the CliffCircular environment, run 
```
python -m sril.cliff_gym_trainer
```

- To train PPO+DynamicBC agent in the Unity riverine environment, run 
```
python -m sril.unity_gym_trainer
```

- To check the training log, run
`tensorboard --logdir sril/ppo_river_tensorboard/` or `tensorboard --logdir sril/ppo_cliff_tensorboard/`





