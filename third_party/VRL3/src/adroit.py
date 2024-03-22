# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# NOTE: adroit env code is currently being cleaned up


from collections import deque
from typing import Any, NamedTuple
import warnings

import dm_env
import numpy as np
from dm_env import StepType, specs
from collections import OrderedDict
import mj_envs
# import adept_envs # TODO worry about this later
import gym

from mjrl.utils.gym_env import GymEnv
# from rrl_local.rrl_utils import make_basic_env, make_dir
from rrl_local.rrl_multicam import BasicAdroitEnv, BasicFrankaEnv

# similar to dmc.py, we will have environment wrapper here...

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)

class ExtendedTimeStepAdroit(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    observation_sensor: Any
    action: Any
    n_goal_achieved: Any
    time_limit_reached: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make_basic_env(env, cam_list=[], from_pixels=False, hybrid_state=None, test_image=False, channels_first=False,
    num_repeats=1, num_frames=1):
    e = GymEnv(env)
    env_kwargs = None
    if from_pixels : # TODO might want to improve this part
        height = 84
        width = 84
        latent_dim = height*width*len(cam_list)*3
        # RRL class instance is environment wrapper...
        e = BasicAdroitEnv(e, cameras=cam_list,
            height=height, width=width, latent_dim=latent_dim, hybrid_state=hybrid_state, 
            test_image=test_image, channels_first=channels_first, num_repeats=num_repeats, num_frames=num_frames)
        env_kwargs = {'rrl_kwargs' : e.env_kwargs}
    # if not from pixels... then it's simpler
    return e, env_kwargs

class AdroitEnv:
    # a wrapper class that will make Adroit env looks like a dmc env
    def __init__(self, env_name, test_image=False, cam_list=None,
        num_repeats=2, num_frames=3, env_feature_type='pixels', device=None, reward_rescale=False): 
        default_env_to_cam_list = {
            'hammer-v0': ['top'],
            'door-v0': ['top'],
            'pen-v0': ['vil_camera'],
            'relocate-v0': ['cam1', 'cam2', 'cam3',],
        }
        if cam_list is None:
            cam_list = default_env_to_cam_list[env_name]
        self.env_name = env_name
        reward_rescale_dict = {
            'hammer-v0': 1/100,
            'door-v0': 1/20,
            'pen-v0': 1/50,
            'relocate-v0': 1/30,
        }
        if reward_rescale:
            self.reward_rescale_factor = reward_rescale_dict[env_name]
        else:
            self.reward_rescale_factor = 1

        # env, _ = make_basic_env(env_name, cam_list=cam_list, from_pixels=from_pixels, hybrid_state=True, 
        #     test_image=test_image, channels_first=True, num_repeats=num_repeats, num_frames=num_frames)
        env = GymEnv(env_name)
        if env_feature_type == 'state':
            raise NotImplementedError("state env not ready")
        elif env_feature_type == 'resnet18' or env_feature_type == 'resnet34' :
            # TODO maybe we will just throw everything into it.. 
            height = 256
            width = 256
            latent_dim = 512
            env = BasicAdroitEnv(env, cameras=cam_list,
                height=height, width=width, latent_dim=latent_dim, hybrid_state=True, 
                test_image=test_image, channels_first=False, num_repeats=num_repeats, num_frames=num_frames, encoder_type=env_feature_type, 
                device=device
                )
        elif env_feature_type == 'pixels':
            height = 84
            width = 84
            latent_dim = height*width*len(cam_list)*num_frames
            # RRL class instance is environment wrapper...
            env = BasicAdroitEnv(env, cameras=cam_list,
                height=height, width=width, latent_dim=latent_dim, hybrid_state=True, 
                test_image=test_image, channels_first=True, num_repeats=num_repeats, num_frames=num_frames, device=device)
        else:
            raise ValueError("env feature not supported")

        self._env = env
        self.obs_dim = env.spec.observation_dim
        self.obs_sensor_dim = 24
        self.act_dim = env.spec.action_dim
        self.horizon = env.spec.horizon
        number_channel = len(cam_list) * 3 * num_frames

        if env_feature_type == 'pixels':
            self._obs_spec = specs.BoundedArray(shape=(number_channel, 84, 84), dtype='uint8', name='observation', minimum=0, maximum=255)
            self._obs_sensor_spec = specs.Array(shape=(self.obs_sensor_dim,), dtype='float32', name='observation_sensor')
        elif env_feature_type == 'resnet18' or env_feature_type == 'resnet34' :
            self._obs_spec = specs.Array(shape=(512 * num_frames *len(cam_list) ,), dtype='float32', name='observation') # TODO fix magic number 
            self._obs_sensor_spec = specs.Array(shape=(self.obs_sensor_dim,), dtype='float32', name='observation_sensor')
        self._action_spec = specs.BoundedArray(shape=(self.act_dim,), dtype='float32', name='action', minimum=-1.0, maximum=1.0)

    def reset(self):
        # pixels and sensor values
        obs_pixels, obs_sensor = self._env.reset()
        obs_sensor = obs_sensor.astype(np.float32)
        action_spec = self.action_spec()
        action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        time_step = ExtendedTimeStepAdroit(observation=obs_pixels,
                                     observation_sensor=obs_sensor,
                                step_type=StepType.FIRST,
                                action=action,
                                reward=0.0,
                                discount=1.0,
                                n_goal_achieved=0,
                                time_limit_reached=False)
        return time_step

    def get_current_obs_without_reset(self):
        # use this to obtain the first state in a demo
        obs_pixels, obs_sensor = self._env.get_obs_for_first_state_but_without_reset()
        obs_sensor = obs_sensor.astype(np.float32)
        action_spec = self.action_spec()
        action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        time_step = ExtendedTimeStepAdroit(observation=obs_pixels,
                                     observation_sensor=obs_sensor,
                                step_type=StepType.FIRST,
                                action=action,
                                reward=0.0,
                                discount=1.0,
                                n_goal_achieved=0,
                                time_limit_reached=False)
        return time_step

    def get_pixels_with_width_height(self, w, h):
        return self._env.get_pixels_with_width_height(w, h)

    def step(self, action, force_step_type=None, debug=False):
        obs_all, reward, done, env_info = self._env.step(action)
        obs_pixels, obs_sensor = obs_all
        obs_sensor = obs_sensor.astype(np.float32)

        discount = 1.0
        n_goal_achieved = env_info['n_goal_achieved']
        time_limit_reached = env_info['TimeLimit.truncated'] if 'TimeLimit.truncated' in env_info else False
        if done:
            steptype = StepType.LAST
        else:
            steptype = StepType.MID

        if done and not time_limit_reached:
            discount = 0.0

        if force_step_type is not None:
            if force_step_type == 'mid':
                steptype = StepType.MID
            elif force_step_type == 'last':
                steptype = StepType.LAST
            else:
                steptype = StepType.FIRST

        reward = reward * self.reward_rescale_factor

        time_step = ExtendedTimeStepAdroit(observation=obs_pixels,
                                     observation_sensor=obs_sensor,
                                step_type=steptype,
                                action=action,
                                reward=reward,
                                discount=discount,
                                n_goal_achieved=n_goal_achieved,
                                time_limit_reached=time_limit_reached)

        if debug:
            return obs_all, reward, done, env_info
        return time_step

    def observation_spec(self):
        return self._obs_spec

    def observation_sensor_spec(self):
        return self._obs_sensor_spec

    def action_spec(self):
        return self._action_spec

    def set_env_state(self, state):
        self._env.set_env_state(state)
    # def __getattr__(self, name):
    #     return getattr(self, name)

    def get_mujoco_sim(self):
        """
        return the underlying mujoco sim
        """
        return self._env.sim
