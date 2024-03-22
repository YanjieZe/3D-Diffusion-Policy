import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
from collections import OrderedDict
import torch.nn as nn
import argparse
from dexart.env.create_env import create_env
from dexart.env.task_setting import TRAIN_CONFIG, IMG_CONFIG, RANDOM_CONFIG
from stable_baselines3.common.torch_layers import PointNetImaginationExtractorGP
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_3d_policy_kwargs(extractor_name):
    feature_extractor_class = PointNetImaginationExtractorGP
    feature_extractor_kwargs = {"pc_key": "instance_1-point_cloud", "gt_key": "instance_1-seg_gt",
                                "extractor_name": extractor_name,
                                "imagination_keys": [f'imagination_{key}' for key in IMG_CONFIG['robot'].keys()],
                                "state_key": "state"}

    policy_kwargs = {
        "features_extractor_class": feature_extractor_class,
        "features_extractor_kwargs": feature_extractor_kwargs,
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
        "activation_fn": nn.ReLU,
    }
    return policy_kwargs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ep', type=int, default=10)
    parser.add_argument('--bs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--freeze', dest='freeze', action='store_true', default=False)
    parser.add_argument('--task_name', type=str, default="laptop")
    parser.add_argument('--extractor_name', type=str, default="smallpn")
    parser.add_argument('--pretrain_path', type=str, default=None)
    args = parser.parse_args()

    task_name = args.task_name
    extractor_name = args.extractor_name
    seed = args.seed if args.seed >= 0 else random.randint(0, 100000)
    pretrain_path = args.pretrain_path
    horizon = 200
    env_iter = args.iter * horizon * args.n
    print(f"freeze: {args.freeze}")

    rand_pos = RANDOM_CONFIG[task_name]['rand_pos']
    rand_degree = RANDOM_CONFIG[task_name]['rand_degree']


    def create_env_fn():
        seen_indeces = TRAIN_CONFIG[task_name]['seen']
        environment = create_env(task_name=task_name,
                                 use_visual_obs=True,
                                 use_gui=False,
                                 is_eval=False,
                                 pc_noise=True,
                                 index=seen_indeces,
                                 img_type='robot',
                                 rand_pos=rand_pos,
                                 rand_degree=rand_degree
                                 )
        return environment


    def create_eval_env_fn():
        unseen_indeces = TRAIN_CONFIG[task_name]['unseen']
        environment = create_env(task_name=task_name,
                                 use_visual_obs=True,
                                 use_gui=False,
                                 is_eval=True,
                                 pc_noise=True,
                                 index=unseen_indeces,
                                 img_type='robot',
                                 rand_pos=rand_pos,
                                 rand_degree=rand_degree)
        return environment


    env = SubprocVecEnv([create_env_fn] * args.workers, "spawn")  # train on a list of envs.

    model = PPO("PointCloudPolicy", env, verbose=1,
                n_epochs=args.ep,
                n_steps=(args.n // args.workers) * horizon,
                learning_rate=args.lr,
                batch_size=args.bs,
                seed=seed,
                policy_kwargs=get_3d_policy_kwargs(extractor_name=extractor_name),
                min_lr=args.lr,
                max_lr=args.lr,
                adaptive_kl=0.02,
                target_kl=0.2,
                )

    if pretrain_path is not None:
        state_dict: OrderedDict = torch.load(pretrain_path)
        model.policy.features_extractor.extractor.load_state_dict(state_dict, strict=False)
        print("load pretrained model: ", pretrain_path)

    rollout = int(model.num_timesteps / (horizon * args.n))

    # after loading or init the model, then freeze it if needed
    if args.freeze:
        model.policy.features_extractor.extractor.eval()
        for param in model.policy.features_extractor.extractor.parameters():
            param.requires_grad = False
        print("freeze model!")

    model.learn(
        total_timesteps=int(env_iter),
        reset_num_timesteps=False,
        iter_start=rollout,
        callback=None
    )
