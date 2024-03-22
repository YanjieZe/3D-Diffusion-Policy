# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur
# Copyright (c) Facebook, Inc. and its affiliates

import gym
from abc import ABC
import numpy as np
from rrl.encoder import Encoder, IdentityEncoder
from PIL import Image
import numpy as np
import torch

_mj_envs = {'pen-v0', 'hammer-v0', 'door-v0', 'relocate-v0'}

def make_encoder(encoder, encoder_type, device, is_eval=True) :
    if not encoder :
        if encoder_type == 'resnet34' :
            encoder = Encoder(encoder_type)
        elif encoder_type == 'identity' :
            encoder = IdentityEncoder()
        else :
            print("Please enter valid encoder_type.")
            raise Exception
    if is_eval:
        encoder.eval()
    encoder.to(device)
    return encoder

class RRL(gym.Env, ABC):
    def __init__(self, env, cameras, encoder_type="resnet34", encoder=None, latent_dim=512, hybrid_state=True, channels_first=False, height=100, width=100, device_id=0):
        num_gpu = torch.cuda.device_count()
        device_id = device_id % num_gpu
        self._env = env
        self.env_id = env.env.unwrapped.spec.id

        self.cameras = cameras
        self.encoder_type = encoder_type
        self.latent_dim = latent_dim
        self.hybrid_state = hybrid_state
        self.channels_first = channels_first
        self.height = height
        self.width = width
        self.action_space = self._env.action_space
        self.device_id = device_id
        self.env_kwargs = {'cameras' : cameras, 'encoder_type': encoder_type, 'encoder': encoder, 'latent_dim' : latent_dim, 'hybrid_state': hybrid_state, 'channels_first' : channels_first, 'height' : height, 'width' : width}

        shape = [latent_dim]
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )
        self.sim = env.env.sim

        self._env.spec.observation_dim = latent_dim
        if hybrid_state :
            if self.env_id in _mj_envs:
                self._env.spec.observation_dim += 24 # Assuming 24 states for adroit hand.

        self.device = "cuda:"+str(device_id)
        self.encoder = make_encoder(encoder=encoder, encoder_type=self.encoder_type, device=self.device, is_eval=True)
        self.transforms = self.encoder.get_transform()
        self.spec = self._env.spec
        self.observation_dim = self.spec.observation_dim
        self.horizon = self._env.env.spec.max_episode_steps

    def get_obs(self, state):
        if self.env_id in _mj_envs :
            env_state = self._env.env.get_env_state()
            qp = env_state['qpos']

        if self.env_id == 'pen-v0':
            qp = qp[:-6]
        elif self.env_id == 'door-v0':
            qp = qp[4:-2]
        elif self.env_id == 'hammer-v0':
            qp = qp[2:-7]
        elif self.env_id == 'relocate-v0':
            qp = qp[6:-6]

        imgs = []
        for cam in self.cameras :
            img = self._env.env.sim.render(width=self.width, height=self.height, mode='offscreen', camera_name=cam, device_id=self.device_id)
            img = img[::-1, :, : ] # Image given has to be flipped
            if self.channels_first :
                img = img.transpose((2, 0, 1))
            #img = img.astype(np.uint8)
            img = Image.fromarray(img)
            img = self.transforms(img)
            imgs.append(img)

        inp_img = torch.stack(imgs).to(self.device) # [num_cam, C, H, W]
        z = self.encoder.get_features(inp_img).reshape(-1)
        assert z.shape[0] == self.latent_dim, "Encoded feature length : {}, Expected : {}".format(z.shape[0], self.latent_dim)

        if self.hybrid_state :
            z = np.hstack((z,qp))
        return z

    def get_env_infos(self):
        return self._env.get_env_infos()
    def set_seed(self, seed):
        return self._env.set_seed(seed)

    def reset(self):
        obs = self._env.reset()
        obs = self.get_obs(obs)
        return obs

    def step(self, action):
        obs, reward, done, env_info = self._env.step(action)
        obs = self.get_obs(obs)
        return obs, reward, done, env_info

    def set_env_state(self, state):
        return self._env.set_env_state(state)
    def get_env_state(self):
        return self._env.get_env_state(state)

    def evaluate_policy(self, policy,
    					num_episodes=5,
    					horizon=None,
    					gamma=1,
    					visual=False,
    					percentile=[],
    					get_full_dist=False,
    					mean_action=False,
    					init_env_state=None,
    					terminate_at_done=True,
    					seed=123):

        self.set_seed(seed)
        horizon = self.horizon if horizon is None else horizon
        mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
        ep_returns = np.zeros(num_episodes)
        self.encoder.eval()

        for ep in range(num_episodes):
            o = self.reset()
            if init_env_state is not None:
                self.set_env_state(init_env_state)
            t, done = 0, False
            while t < horizon and (done == False or terminate_at_done == False):
                self.render() if visual is True else None
                o = self.get_obs(self._env.get_obs())
                a = policy.get_action(o)[1]['evaluation'] if mean_action is True else policy.get_action(o)[0]
                o, r, done, _ = self.step(a)
                ep_returns[ep] += (gamma ** t) * r
                t += 1

        mean_eval, std = np.mean(ep_returns), np.std(ep_returns)
        min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
        base_stats = [mean_eval, std, min_eval, max_eval]

        percentile_stats = []
        for p in percentile:
            percentile_stats.append(np.percentile(ep_returns, p))

        full_dist = ep_returns if get_full_dist is True else None

        return [base_stats, percentile_stats, full_dist]

