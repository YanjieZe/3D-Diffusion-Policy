import json
import os
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
import sapien.core as sapien
import transforms3d
from sapien.utils import Viewer

from dexart.env.rl_env.base import BaseRLEnv
from dexart.env.sim_env.constructor import add_default_scene_light
from dexart.env.sim_env.faucet_env import FaucetEnv

class FaucetRLEnv(FaucetEnv, BaseRLEnv):
    def __init__(self, use_gui=False, frame_skip=5, robot_name="adroit_hand_free", friction=5, index=0, rand_pos=0.0,
                 rand_orn=0.0, **renderer_kwargs):
        super().__init__(use_gui, frame_skip, friction=5, index=index, **renderer_kwargs)
        # ============== status definition ==============
        self.instance_init_pos = None
        self.robot_init_pose = None
        self.robot_object_contact = None
        self.finger_tip_pos = None
        self.rand_pos = rand_pos
        self.rand_orn = rand_orn
        
        self.target_openness = 1.2 # originally: 1.5
        # ============== will not change during training and randomize instance ==============
        self.robot_name = robot_name
        self.setup(robot_name)
        self.robot_init_pose = sapien.Pose(np.array([-0.5, 0, 0]), transforms3d.euler.euler2quat(0, 0, 0))
        self.robot.set_pose(self.robot_init_pose)
        self.configure_robot_contact_reward()
        self.robot_annotation = self.setup_robot_annotation(robot_name)
        # ============== will change if randomize instance ==============
        self.reset()

    def update_cached_state(self):
        for i, link in enumerate(self.finger_tip_links):
            self.finger_tip_pos[i] = self.finger_tip_links[i].get_pose().p
        check_contact_links = self.finger_contact_links + [self.palm_link]
        finger_contact_boolean = self.check_actor_pair_contacts(check_contact_links, self.handle_link)
        self.robot_object_contact[:] = np.clip(np.bincount(self.finger_contact_ids, weights=finger_contact_boolean), 0,
                                               1)
        arm_contact_boolean = self.check_actors_pair_contacts(self.arm_contact_links, self.instance_links)
        self.is_arm_contact = np.sum(arm_contact_boolean)
        self.robot_qpos_vec = self.robot.get_qpos()
        self.openness = abs(self.instance.get_qpos()[self.revolute_joint_index])
        self.handle_pose = self.handle_link.get_pose()
        self.palm_pose = self.palm_link.get_pose()
        trans_matrix = self.palm_pose.to_transformation_matrix()
        self.palm_vector = trans_matrix[:3, :3] @ np.array([1, 0, 0])

        self.handle_in_palm = self.handle_pose.p - self.palm_pose.p
        self.palm_v = self.palm_link.get_velocity()
        self.palm_w = self.palm_link.get_angular_velocity()

        self.is_contact = np.sum(self.robot_object_contact[:-1]) >= 2 and self.robot_object_contact[-1]
        if np.linalg.norm(self.palm_pose.p - self.handle_pose.p) > 0.2:  # Reaching
            self.state = 1
        elif not self.is_contact:  # Not yet firmly grasped. So go to stage 2, grasping.
            self.state = 2
        else:  # Fully grasped. So go to stage 3, turning on.
            self.state = 3
        self.early_done = (self.openness > self.target_openness) and (self.state == 3)
        self.is_eval_done = (self.openness > self.target_openness) and (self.state == 3)

    def get_oracle_state(self):
        return np.concatenate([
            self.robot_qpos_vec, self.palm_v, self.palm_w, self.palm_pose.p,
            [float(self.current_step) / float(self.horizon)]
        ])

    def get_robot_state(self):
        return np.concatenate([
            self.robot_qpos_vec, self.palm_v, self.palm_w, self.palm_pose.p,
            [float(self.current_step) / float(self.horizon)]
        ])

    def get_reward(self, action):
        reward = 0
        if self.state == 1:
            reward = -0.1 * min(np.linalg.norm(self.palm_pose.p - self.handle_pose.p),
                                0.5)  # encourage palm be close to handle
        elif self.state == 2:
            reward += 0.2 * (int(self.is_contact))
            reward -= 0.1 * (int(self.is_arm_contact))
        elif self.state == 3:
            reward += 0.2 * (int(self.is_contact))
            reward -= 0.1 * (int(self.is_arm_contact))
            reward += 1.0 * self.openness
        if self.early_done:
            reward += (self.horizon - self.current_step) * 1.2 * self.openness
        action_penalty = np.sum(np.clip(self.robot.get_qvel(), -1, 1) ** 2) * 0.01
        controller_penalty = (self.cartesian_error ** 2) * 1e3
        reward -= 0.01 * (action_penalty + controller_penalty)
        return reward

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        # reset status
        self.robot.set_pose(self.robot_init_pose)
        # reset changeable status if randomize instance
        self.reset_internal()  # change instance if randomize instance
        if self.need_flush_when_change_instance and self.change_instance_when_reset:
            self.flush_imagination_config()
        if self.robot_annotation.__contains__(str(self.index)):
            self.instance_init_pos = self.robot.get_pose().p + np.array(self.robot_annotation[str(self.index)])
        else:
            self.instance_init_pos = self.pos
        self.pos = self.instance_init_pos
        pos = self.pos + np.random.random(3) * self.rand_pos  # can add noise here to randomize loaded position
        random_orn = (np.random.rand() * 2 - 1) * self.rand_orn
        orn = transforms3d.euler.euler2quat(0, 0, random_orn)
        self.instance.set_root_pose(sapien.Pose(pos, orn))
        self.update_cached_state()
        self.update_imagination(reset_goal=False)
        return self.get_observation()

    def setup_robot_annotation(self, robot_name: str):
        # here we load robot2faucet
        current_dir = Path(__file__).parent
        self.pos_path = current_dir.parent.parent.parent / "assets" / "annotation"/ f"faucet_{robot_name}_relative_position.json"
        if not os.path.exists(self.pos_path):
            raise FileNotFoundError
        else:
            with open(self.pos_path, "r") as f:
                pos_dict = json.load(f)
            return pos_dict

    @cached_property
    def obs_dim(self):
        if not self.use_visual_obs:
            return len(self.get_oracle_state())
        else:
            return len(self.get_robot_state())

    def is_done(self):
        return (self.current_step >= self.horizon) or self.early_done

    @cached_property
    def horizon(self):
        return 250