# DexArt: Benchmarking Generalizable Dexterous Manipulation with Articulated Objects

[[Project Page]](https://www.chenbao.tech/dexart/) [[arXiv]](https://arxiv.org/abs/2305.05706) [[Paper]](https://www.chenbao.tech/dexart/static/paper/dexart.pdf)
-----

[DexArt: Benchmarking Generalizable Dexterous Manipulation with Articulated Objects](https://www.chenbao.tech/dexart/), 


[Chen Bao](https://chenbao.tech)\*, [Helin Xu](https://helinxu.github.io/)\*, [Yuzhe Qin](https://yzqin.github.io/), [Xiaolong Wang](https://xiaolonw.github.io/), CVPR 2023.


DexArt is a novel benchmark and pipeline for learning multiple dexterous manipulation tasks.
This repo contains the **simulated environment** and **training code** for DexArt.

![DexArt Teaser](docs/teaser.png)


## Installation

1. Clone the repo and Create a conda env with all the Python dependencies.

```bash
git clone git@github.com:Kami-code/dexart-release.git
cd dexart-release
conda create --name dexart python=3.8
conda activate dexart
pip install -e .    # for simulation environment
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch    # for visualizing trained policy and training 
```

2. Download the assets from
the [Google Drive](https://drive.google.com/file/d/1JdReXZjMaqMO0HkZQ4YMiU2wTdGCgum1/view?usp=sharing) and place 
the `asset` directory at the project root directory.

## File Structure
The file structure is listed as follows:

`dexart/env/`: environments

`assets/`: tasks annotations, object and robot URDFs

`examples/`: example code to try DexArt

`stable_baselines3/`: rl training code modified from [stable_baselines3](https://github.com/DLR-RM/stable-baselines3)



## Quick Start

### Example of Random Action


```bash
python examples/random_action.py --task_name=laptop
```

`task_name`: name of environment [`faucet`, `laptop`, `bucket`, `toilet`]

### Example for Visualizing Point Cloud Observation 

```bash
python examples/visualize_observation.py --task_name=laptop
```
`task_name`: name of environment [`faucet`, `laptop`, `bucket`, `toilet`]


### Example for Visualizing Policy

```bash
python examples/visualize_policy.py --task_name=laptop --checkpoint_path assets/rl_checkpoints/laptop.zip
```

`task_name`: name of environment [`faucet`, `laptop`, `bucket`, `toilet`]

`use_test_set`: flag to determine evaluating with seen or unseen instances

### Example for Evaluating Policy

```bash
python examples/evaluate_policy.py --task_name=laptop --checkpoint_path assets/rl_checkpoints/laptop.zip --eval_per_instance 10
```

`task_name`: name of environment [`faucet`, `laptop`, `bucket`, `toilet`]

`use_test_set`: flag to determine evaluating with seen or unseen instances

### Example for Training RL Agent

```bash
python3 examples/train.py --n 100 --workers 10 --iter 5000 --lr 0.0001 &&
--seed 100 --bs 500 --task_name laptop --extractor_name smallpn &&
--pretrain_path ./assets/vision_pretrain/laptop_smallpn_fulldata.pth 
```
`n`: the number of rollouts to be collected in single episode

`workers`: the number of simulation progress

`iter`: the total episode number to be trained

`lr`: learning rate of RL

`seed`: seed of RL

`bs`: batch size of RL update

`task_name`: name of training environment [`faucet`, `laptop`, `bucket`, `toilet`]

`extractor_name`: different PointNet architectures [`smallpn`, `meduimpn`, `largepn`]

`pretrain_path`: path to downloaded pretrained model. [Default: `None`]

## Bibtex

```
@inproceedings{
    bao2023dexart,
    title={DexArt: Benchmarking Generalizable Dexterous Manipulation with Articulated Objects},
    author={Chen Bao and Helin Xu and Yuzhe Qin and Xiaolong Wang},
    booktitle={Conference on Computer Vision and Pattern Recognition 2023},
    year={2023},
    url={https://openreview.net/forum?id=v-KQONFyeKp}
}
```

## Acknowledgements

This repository employs the same code structure for simulation environment and training code to that used in [DexPoint](https://github.com/yzqin/dexpoint-release).
