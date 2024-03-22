# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur
# Copyright (c) Facebook, Inc. and its affiliates

import logging
logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch
import torch.nn as nn
from torchvision import models, transforms
from rrl.encoder import Encoder
from PIL import Image

import mj_envs
import mjrl
import gym
from mjrl.utils.gym_env import GymEnv
from rrl.utils import make_env


if __name__ == '__main__':
    env_name = 'hammer-v0'
    cam = "left_cross"
    device_id = 0
    encoder_type = 'resnet34'
    #encoder_type = 'identity'
    #encoder_type = None

    #env = GymEnv(env_name)
    env, _ = make_env(env_name, from_pixels=True, cam_list=[cam], encoder_type=encoder_type, hybrid_state=True)
    z = env.reset()
    print(z.shape)

