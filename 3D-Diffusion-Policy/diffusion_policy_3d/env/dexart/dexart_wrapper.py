import gym
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops

from termcolor import cprint
from dexart.env.task_setting import TRAIN_CONFIG, RANDOM_CONFIG
from dexart.env.create_env import create_env
from gym import spaces


def downsample_with_fps(points: np.ndarray, num_points: int = 1024):
    # fast point cloud sampling using torch3d
    points = torch.from_numpy(points).unsqueeze(0).cuda()
    num_points = torch.tensor([num_points]).cuda()
    # remember to only use coord to sample
    _, sampled_indices = torch3d_ops.sample_farthest_points(points=points[...,:3], K=num_points)
    points = points.squeeze(0).cpu().numpy()
    points = points[sampled_indices.squeeze(0).cpu().numpy()]
    return points


class DexArtEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task_name, use_test_set=False, num_points=1024):
        if use_test_set:
            indeces = TRAIN_CONFIG[task_name]['unseen']
            cprint(f"using unseen instances {indeces}", 'yellow')
        else:
            indeces = TRAIN_CONFIG[task_name]['seen']
            cprint(f"using seen instances {indeces}", 'yellow')

        rand_pos = RANDOM_CONFIG[task_name]['rand_pos']
        rand_degree = RANDOM_CONFIG[task_name]['rand_degree']

        self.env = create_env(task_name=task_name,
                              use_visual_obs=True,
                              use_gui=False,
                              is_eval=True,
                              pc_noise=True,
                              pc_seg=True,
                              index=indeces,
                              img_type='robot',
                              rand_pos=rand_pos,
                              rand_degree=rand_degree)

        robot_dof = self.env.robot.dof
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(robot_dof,),
            dtype=np.float32
        )
        self.obs_sensor_dim = 32
        self.num_points = num_points
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3, 84, 84),
                dtype=np.float32
            ),
            
            'depth': spaces.Box(
                low=0,
                high=1,
                shape=(84, 84),
                dtype=np.float32
            ),
            
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_sensor_dim,),
                dtype=np.float32
            ),
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 3),
                dtype=np.float32
            ),
            'imagin_robot': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(96, 7),
                dtype=np.float32
            ),

        })

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_pixels = obs['instance_1-rgb']  # (84, 84, 3)
        obs_depth = obs['instance_1-depth']  # (84, 84)
        obs_sensor = obs['state']  # (32,)
        obs_pointcloud = obs['instance_1-point_cloud']  # (1024, 3)
        if obs_pointcloud.shape[0] > self.num_points:
            obs_pointcloud = downsample_with_fps(
                obs_pointcloud, self.num_points)
        obs_imagin_robot = obs['imagination_robot']  # (96, 7)

        if obs_pixels.shape[0] != 3:  # make channel first
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': obs_depth,
            'agent_pos': obs_sensor,
            'point_cloud': obs_pointcloud,
            'imagin_robot': obs_imagin_robot
        }
        return obs_dict, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs_pixels = obs['instance_1-rgb']  # (84, 84, 3)
        obs_sensor = obs['state']  # (32,)
        obs_pointcloud = obs['instance_1-point_cloud']  # (1024, 3)
        obs_depth = obs['instance_1-depth']  # (84, 84)
        if obs_pointcloud.shape[0] > self.num_points:
            obs_pointcloud = downsample_with_fps(
                obs_pointcloud, self.num_points)
        obs_imagin_robot = obs['imagination_robot']  # (96, 7)

        if obs_pixels.shape[0] != 3:  # make channel first
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': obs_depth,
            'agent_pos': obs_sensor,
            'point_cloud': obs_pointcloud,
            'imagin_robot': obs_imagin_robot
        }
        return obs_dict

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def get_visual_observation(self):
        return self.env.get_visual_observation()

    def render(self, mode='rgb_array'):
        visual_obs = self.get_visual_observation()
        img = visual_obs['instance_1-rgb']  # (84,84,3), [0,1]
        # to uint8
        img = (img*255).astype(np.uint8)
        return img

    def close(self):
        pass

    def horizon(self):
        return self.env.horizon()

    def is_success(self):
        return self.env.is_eval_done
