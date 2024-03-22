# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur
# Copyright (c) Facebook, Inc. and its affiliates

import gym
# from abc import ABC
import numpy as np
from .rrl_encoder import Encoder, IdentityEncoder
from PIL import Image
import torch
from collections import deque

_mj_envs = {'pen-v0', 'hammer-v0', 'door-v0', 'relocate-v0'}

def make_encoder(encoder, encoder_type, device, is_eval=True) :
    if not encoder :
        if encoder_type == 'resnet34' or encoder_type == 'resnet18' :
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

class BasicAdroitEnv(gym.Env): # , ABC
    def __init__(self, env, cameras, latent_dim=512, hybrid_state=True, channels_first=False, 
    height=84, width=84, test_image=False, num_repeats=1, num_frames=1, encoder_type=None, device=None):
        self._env = env
        self.env_id = env.env.unwrapped.spec.id
        self.device = device

        self._num_repeats = num_repeats
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)

        self.encoder = None
        self.transforms = None
        self.encoder_type = encoder_type
        if encoder_type is not None:
            self.encoder = make_encoder(encoder=None, encoder_type=self.encoder_type, device=self.device, is_eval=True)
            self.transforms = self.encoder.get_transform()

        if test_image:
            print("======================adroit image test mode==============================")
            print("======================adroit image test mode==============================")
            print("======================adroit image test mode==============================")
            print("======================adroit image test mode==============================")
        self.test_image = test_image

        self.cameras = cameras
        self.latent_dim = latent_dim
        self.hybrid_state = hybrid_state
        self.channels_first = channels_first
        self.height = height
        self.width = width
        self.action_space = self._env.action_space
        self.env_kwargs = {'cameras' : cameras, 'latent_dim' : latent_dim, 'hybrid_state': hybrid_state,
                           'channels_first' : channels_first, 'height' : height, 'width' : width}

        shape = [3, self.width, self.height]
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )
        self.sim = env.env.sim
        self._env.spec.observation_dim = latent_dim

        if hybrid_state :
            if self.env_id in _mj_envs:
                self._env.spec.observation_dim += 24 # Assuming 24 states for adroit hand.

        self.spec = self._env.spec
        self.observation_dim = self.spec.observation_dim
        self.horizon = self._env.env.spec.max_episode_steps

    def get_obs(self,):
        # for our case, let's output the image, and then also the sensor features
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

        imgs = [] # number of image is number of camera

        if self.encoder is not None:
            for cam in self.cameras :
                img = self._env.env.sim.render(width=self.width, height=self.height, mode='offscreen', camera_name=cam, device_id=0)
                # img = env.env.sim.render(width=84, height=84, mode='offscreen')
                img = img[::-1, :, : ] # Image given has to be flipped
                if self.channels_first :
                    img = img.transpose((2, 0, 1))
                #img = img.astype(np.uint8)
                img = Image.fromarray(img)
                img = self.transforms(img)
                imgs.append(img)

            inp_img = torch.stack(imgs).to(self.device) # [num_cam, C, H, W]
            z = self.encoder.get_features(inp_img).reshape(-1)
            # assert z.shape[0] == self.latent_dim, "Encoded feature length : {}, Expected : {}".format(z.shape[0], self.latent_dim)
            pixels = z
        else:
            if not self.test_image:
                for cam in self.cameras : # for each camera, render once
                    img = self._env.env.sim.render(width=self.width, height=self.height, mode='offscreen', camera_name=cam, device_id=0) # TODO device id will think later
                    # img = img[::-1, :, : ] # Image given has to be flipped
                    if self.channels_first :
                        img = img.transpose((2, 0, 1)) # then it's 3 x width x height
                    # we should do channels first... (not sure why by default it's not, maybe they did some transpose when using the encoder?)
                    #img = img.astype(np.uint8)
                    # img = Image.fromarray(img) # TODO is this necessary?
                    imgs.append(img)
            else:
                img = (np.random.rand(1, 84, 84) * 255).astype(np.uint8)
                imgs.append(img)
            pixels = np.concatenate(imgs, axis=0)

        # TODO below are what we originally had... 
        # if not self.test_image:
        #     for cam in self.cameras : # for each camera, render once
        #         img = self._env.env.sim.render(width=self.width, height=self.height, mode='offscreen', camera_name=cam, device_id=0) # TODO device id will think later
        #         # img = img[::-1, :, : ] # Image given has to be flipped
        #         if self.channels_first :
        #             img = img.transpose((2, 0, 1)) # then it's 3 x width x height
        #         # we should do channels first... (not sure why by default it's not, maybe they did some transpose when using the encoder?)
        #         #img = img.astype(np.uint8)
        #         # img = Image.fromarray(img) # TODO is this necessary?
        #         imgs.append(img)
        # else:
        #     img = (np.random.rand(1, 84, 84) * 255).astype(np.uint8)
        #     imgs.append(img)
        # pixels = np.concatenate(imgs, axis=0)

        if not self.hybrid_state : # this defaults to True... so RRL uses hybrid state
            qp = None

        sensor_info = qp
        return pixels, sensor_info

    def get_env_infos(self):
        return self._env.get_env_infos()
    
    def set_seed(self, seed):
        return self._env.set_seed(seed)

    def get_stacked_pixels(self): #TODO fix it
        assert len(self._frames) == self._num_frames
        stacked_pixels = np.concatenate(list(self._frames), axis=0)
        return stacked_pixels

    def reset(self):
        self._env.reset()
        pixels, sensor_info = self.get_obs()
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        stacked_pixels = self.get_stacked_pixels()
        return stacked_pixels, sensor_info

    def get_obs_for_first_state_but_without_reset(self):
        pixels, sensor_info = self.get_obs()
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        stacked_pixels = self.get_stacked_pixels()
        return stacked_pixels, sensor_info

    def step(self, action):
        reward_sum = 0.0
        discount_prod = 1.0 # TODO pen can terminate early 
        n_goal_achieved = 0
        for i_action in range(self._num_repeats): 
            obs, reward, done, env_info = self._env.step(action)
            reward_sum += reward 
            if env_info['goal_achieved'] == True:
                n_goal_achieved += 1
            if done:
                break
        env_info['n_goal_achieved'] = n_goal_achieved
        # now get stacked frames
        pixels, sensor_info = self.get_obs()
        self._frames.append(pixels)
        stacked_pixels = self.get_stacked_pixels()
        return [stacked_pixels, sensor_info], reward_sum, done, env_info

    def set_env_state(self, state):
        return self._env.set_env_state(state)
    
    def get_env_state(self):
        return self._env.get_env_state()

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
        # TODO this needs to be rewritten

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

    def get_pixels_with_width_height(self, w, h):
        imgs = [] # number of image is number of camera

        for cam in self.cameras : # for each camera, render once
            img = self._env.env.sim.render(width=w, height=h, mode='offscreen', camera_name=cam, device_id=0) # TODO device id will think later
            # img = img[::-1, :, : ] # Image given has to be flipped
            if self.channels_first :
                img = img.transpose((2, 0, 1)) # then it's 3 x width x height
            # we should do channels first... (not sure why by default it's not, maybe they did some transpose when using the encoder?)
            #img = img.astype(np.uint8)
            # img = Image.fromarray(img) # TODO is this necessary?
            imgs.append(img)

        pixels = np.concatenate(imgs, axis=0)
        return pixels


class BasicFrankaEnv(gym.Env):
    def __init__(self, env, cameras, latent_dim=512, hybrid_state=True, channels_first=False,
    height=84, width=84, test_image=False, num_repeats=1, num_frames=1, encoder_type=None, device=None):
        # the parameter env is basically the kitchen env now
        self._env = env
        self.env_id = env.env.unwrapped.spec.id
        self.device = device

        self._num_repeats = num_repeats
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)

        self.encoder = None
        self.transforms = None
        self.encoder_type = encoder_type
        if encoder_type is not None:
            self.encoder = make_encoder(encoder=None, encoder_type=self.encoder_type, device=self.device, is_eval=True)
            self.transforms = self.encoder.get_transform()

        if test_image:
            print("======================adroit image test mode==============================")
            print("======================adroit image test mode==============================")
            print("======================adroit image test mode==============================")
            print("======================adroit image test mode==============================")
        self.test_image = test_image

        self.cameras = cameras
        self.latent_dim = latent_dim
        self.hybrid_state = hybrid_state
        self.channels_first = channels_first
        self.height = height
        self.width = width
        self.action_space = self._env.action_space
        self.env_kwargs = {'cameras' : cameras, 'latent_dim' : latent_dim, 'hybrid_state': hybrid_state,
                           'channels_first' : channels_first, 'height' : height, 'width' : width}

        shape = [3, self.width, self.height]
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )
        self.sim = env.env.sim
        self._env.spec.observation_dim = latent_dim
        self._env.spec.action_dim = 9 # TODO magic number
        self._env.spec.horizon = self._env.env.spec.max_episode_steps
        # print("==============")
        # print(dir(self._env))
        # print("high: ", self._env.action_space)
        # quit()

        if hybrid_state :
            if self.env_id in _mj_envs:
                self._env.spec.observation_dim += 24 # Assuming 24 states for adroit hand.

        self.spec = self._env.spec
        self.observation_dim = self.spec.observation_dim
        self.horizon = self._env.env.spec.max_episode_steps

    def get_obs(self,):
        # for our case, let's output the image, and then also the sensor features
        imgs = [] # number of image is number of camera

        if self.encoder is not None:
            for cam in self.cameras :
                # img = self._env.env.sim.render(width=self.width, height=self.height, mode='offscreen', camera_name=cam, device_id=0)
                img = self._env.env.sim.render(width=84, height=84)
                # img = env.env.sim.render(width=84, height=84, mode='offscreen')
                img = img[::-1, :, : ] # Image given has to be flipped
                if self.channels_first :
                    img = img.transpose((2, 0, 1))
                #img = img.astype(np.uint8)
                img = Image.fromarray(img)
                img = self.transforms(img)
                imgs.append(img)

            inp_img = torch.stack(imgs).to(self.device) # [num_cam, C, H, W]
            z = self.encoder.get_features(inp_img).reshape(-1)
            # assert z.shape[0] == self.latent_dim, "Encoded feature length : {}, Expected : {}".format(z.shape[0], self.latent_dim)
            pixels = z
        else:
            if not self.test_image:
                for cam in self.cameras : # for each camera, render once
                    img = self._env.env.sim.render(width=84, height=84)
                    # img = self._env.env.sim.render(width=self.width, height=self.height, mode='offscreen', camera_name=cam, device_id=0) # TODO device id will think later
                    # img = img[::-1, :, : ] # Image given has to be flipped
                    if self.channels_first :
                        img = img.transpose((2, 0, 1)) # then it's 3 x width x height
                    # we should do channels first... (not sure why by default it's not, maybe they did some transpose when using the encoder?)
                    #img = img.astype(np.uint8)
                    # img = Image.fromarray(img) # TODO is this necessary?
                    imgs.append(img)
            else:
                img = (np.random.rand(1, 84, 84) * 255).astype(np.uint8)
                imgs.append(img)
            pixels = np.concatenate(imgs, axis=0)

        # TODO below are what we originally had...
        # if not self.test_image:
        #     for cam in self.cameras : # for each camera, render once
        #         img = self._env.env.sim.render(width=self.width, height=self.height, mode='offscreen', camera_name=cam, device_id=0) # TODO device id will think later
        #         # img = img[::-1, :, : ] # Image given has to be flipped
        #         if self.channels_first :
        #             img = img.transpose((2, 0, 1)) # then it's 3 x width x height
        #         # we should do channels first... (not sure why by default it's not, maybe they did some transpose when using the encoder?)
        #         #img = img.astype(np.uint8)
        #         # img = Image.fromarray(img) # TODO is this necessary?
        #         imgs.append(img)
        # else:
        #     img = (np.random.rand(1, 84, 84) * 255).astype(np.uint8)
        #     imgs.append(img)
        # pixels = np.concatenate(imgs, axis=0)

        if not self.hybrid_state : # this defaults to True... so RRL uses hybrid state
            qp = None

        sensor_info = qp
        return pixels, sensor_info

    def get_env_infos(self):
        return self._env.get_env_infos()
    def set_seed(self, seed):
        return self._env.set_seed(seed)

    def get_stacked_pixels(self): #TODO fix it
        assert len(self._frames) == self._num_frames
        stacked_pixels = np.concatenate(list(self._frames), axis=0)
        return stacked_pixels

    def reset(self):
        self._env.reset()
        pixels, sensor_info = self.get_obs()
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        stacked_pixels = self.get_stacked_pixels()
        return stacked_pixels, sensor_info

    def get_obs_for_first_state_but_without_reset(self):
        pixels, sensor_info = self.get_obs()
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        stacked_pixels = self.get_stacked_pixels()
        return stacked_pixels, sensor_info

    def step(self, action):
        reward_sum = 0.0
        discount_prod = 1.0 # TODO pen can terminate early
        n_goal_achieved = 0
        for i_action in range(self._num_repeats):
            obs, reward, done, env_info = self._env.step(action)
            reward_sum += reward
            if env_info['goal_achieved'] == True:
                n_goal_achieved += 1
            if done:
                break
        env_info['n_goal_achieved'] = n_goal_achieved
        # now get stacked frames
        pixels, sensor_info = self.get_obs()
        self._frames.append(pixels)
        stacked_pixels = self.get_stacked_pixels()
        return [stacked_pixels, sensor_info], reward_sum, done, env_info

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
        # TODO this needs to be rewritten

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

    def get_pixels_with_width_height(self, w, h):
        imgs = [] # number of image is number of camera

        for cam in self.cameras : # for each camera, render once
            img = self._env.env.sim.render(width=w, height=h, mode='offscreen', camera_name=cam, device_id=0) # TODO device id will think later
            # img = img[::-1, :, : ] # Image given has to be flipped
            if self.channels_first :
                img = img.transpose((2, 0, 1)) # then it's 3 x width x height
            # we should do channels first... (not sure why by default it's not, maybe they did some transpose when using the encoder?)
            #img = img.astype(np.uint8)
            # img = Image.fromarray(img) # TODO is this necessary?
            imgs.append(img)

        pixels = np.concatenate(imgs, axis=0)
        return pixels
