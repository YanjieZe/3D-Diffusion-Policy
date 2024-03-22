import os

import numpy as np

from dexart.env.rl_env.faucet_env import FaucetRLEnv
from dexart.env.rl_env.bucket_env import BucketRLEnv
from dexart.env.rl_env.laptop_env import LaptopRLEnv
from dexart.env.rl_env.toilet_env import ToiletRLEnv
from dexart.env import task_setting
from dexart.env.sim_env.constructor import add_default_scene_light


def create_env(task_name, use_visual_obs, use_gui=False, is_eval=False, pc_seg=False,
               pc_noise=False, index=-1, img_type=None, rand_pos=0.0, rand_degree=0, frame_skip=10, no_rgb=True,
               **kwargs):
    robot_name = "allegro_hand_xarm6_wrist_mounted_face_front"
    rotation_reward_weight = 1
    env_params = dict(robot_name=robot_name, rotation_reward_weight=rotation_reward_weight,
                      use_visual_obs=use_visual_obs, use_gui=use_gui, no_rgb=no_rgb, use_old_api=True,
                      index=index, frame_skip=frame_skip, rand_pos=rand_pos, rand_orn=rand_degree / 180 * np.pi,
                      **kwargs)
    if img_type:
        assert img_type in task_setting.IMG_CONFIG.keys()

    if is_eval:
        env_params["no_rgb"] = False
    env_params["need_offscreen_render"] = True

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    if task_name == 'faucet':
        env = FaucetRLEnv(**env_params, friction=5)
    elif task_name == 'bucket':
        env = BucketRLEnv(**env_params, friction=0)
    elif task_name == 'laptop':
        env = LaptopRLEnv(**env_params, friction=5)
    elif task_name == 'toilet':
        env = ToiletRLEnv(**env_params, friction=5)
    else:
        raise NotImplementedError
    if use_visual_obs:
        current_setting = task_setting.CAMERA_CONFIG[task_name]
        env.setup_camera_from_config(current_setting)
        if pc_seg:
            env.setup_visual_obs_config(task_setting.OBS_CONFIG["instance_pc_seg"])
        elif pc_noise:
            env.setup_visual_obs_config(task_setting.OBS_CONFIG["instance_noise"])
        else:
            env.setup_visual_obs_config(task_setting.OBS_CONFIG["instance"])
        if img_type:
            # Specify imagination
            env.setup_imagination_config(task_setting.IMG_CONFIG[img_type])
    if is_eval:
        env.setup_camera_from_config(task_setting.CAMERA_CONFIG["viz_only"])
        add_default_scene_light(env.scene, env.renderer)


    # flush cache
    env.action_space
    env.observation_space
    return env