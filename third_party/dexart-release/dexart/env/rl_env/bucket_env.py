import json
import os
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
import sapien.core as sapien
import transforms3d

from dexart.env.rl_env.base import BaseRLEnv
from dexart.env.sim_env.bucket_env import BucketEnv


def getAngle(P, Q):
    R = np.dot(P, Q.T)
    theta = (np.trace(R) - 1) / 2
    return np.arccos(np.clip(theta, a_min=-1, a_max=1)) / np.pi  # 0~1


class BucketRLEnv(BucketEnv, BaseRLEnv):
    def __init__(self, use_gui=False, frame_skip=5, robot_name="adroit_hand_free", friction=0, index=0, rand_pos=0.0,
                 rand_orn=0.0, thick_handle=True, **renderer_kwargs):
        # ============== status definition ==============
        self.instance_init_pos = None
        self.robot_init_pose = None
        self.robot_object_contact = None
        self.robot_instance_base_contact = None
        self.rand_pos = rand_pos
        self.rand_orn = rand_orn
        self.grasp_dist = None
        self.palm_height = None
        
        self.target_delta_height = 0.15 # 2023.08.11
        # =================================================
        super().__init__(use_gui, frame_skip, friction=friction, index=index, handle_type='left', fix_root_link=False,
                         thick_handle=thick_handle, **renderer_kwargs)
        self.box = self.create_box(
            sapien.Pose(p=np.array([-0.5, 0., 0.15])),
            half_size=np.array([0.1, 0.2, 0.15]),
            color=[0.2, 0.2, 0.2],
            name='box',
        )
        self.box.lock_motion()

        # ============== will not change during training and randomize instance ==============
        self.robot_name = robot_name
        self.setup(robot_name)
        self.robot_init_pose = sapien.Pose(np.array([-0.5, 0, 0.3]), transforms3d.euler.euler2quat(0, 0, 0))
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
        self.is_arm_contact = np.sum(arm_contact_boolean) >= 1
        self.loosen_contact = np.sum(self.robot_object_contact[:-1]) >= 1 or self.robot_object_contact[-1]
        self.strict_contact = np.sum(self.robot_object_contact[:-1]) >= 3 and self.robot_object_contact[-1]
        self.is_contact = np.sum(self.robot_object_contact[:-1]) >= 3
        self.is_contact_percent = (np.sum(self.robot_object_contact[:1])) / 4
        self.finger_touched_percent = (np.sum(self.robot_object_contact[:])) / 5
        finger_contact_boolean = self.check_actor_pair_contacts(check_contact_links, self.instance_base_link)
        self.robot_instance_base_contact[:] = np.clip(
            np.bincount(self.finger_contact_ids, weights=finger_contact_boolean), 0,
            1)
        self.finger_base_touched_percent = np.sum(self.robot_instance_base_contact) / 5

        self.is_finger_touch_instance_base_percent = np.sum(self.robot_instance_base_contact[:]) / 5

        self.robot_qpos_vec = self.robot.get_qpos()
        openness = abs(self.instance.get_qpos()[0] - self.joint_limits_dict[str(self.index)]['middle'])
        total = abs(self.joint_limits_dict[str(self.index)]['left'] - self.joint_limits_dict[str(self.index)]['middle'])
        self.progress = 1 - openness / total
        self.handle_pose = self.get_handle_global_pose()
        self.palm_pose = self.palm_link.get_pose()

        trans_matrix = self.palm_pose.to_transformation_matrix()
        self.palm_vector = trans_matrix[:3, :3] @ np.array([1, 0, 0])

        self.handle_in_palm = self.handle_pose.p - self.palm_pose.p
        self.palm_v = self.palm_link.get_velocity()
        self.palm_w = self.palm_link.get_angular_velocity()
        self.height = self.instance_base_link.get_pose().p[2]
        self.delta_height = max(self.height - self.init_height, 0)
        self.pose_mat = transforms3d.quaternions.quat2mat(self.instance.get_root_pose().q)
        self.degree_progress = getAngle(self.init_pose_mat, self.pose_mat)
        if np.linalg.norm(self.palm_pose.p - self.handle_pose.p) > 0.2:  # Reaching
            self.state = 1
        elif not self.is_contact or self.progress < 0.5:
            self.state = 2
        else:
            self.state = 3
        self.early_done = (self.progress > 0.9) and (self.state == 3) and (self.delta_height > 0.3)
        self.is_eval_done = (self.progress > 0.7) and (self.delta_height > self.target_delta_height)
        self.last_palm_height = self.palm_height if self.palm_height else self.palm_pose.p[2]
        self.palm_height = self.palm_pose.p[2]
        self.last_grasp_dist = self.grasp_dist if self.grasp_dist else (np.linalg.norm(
            self.finger_tip_pos[0] - self.finger_tip_pos[1]) + np.linalg.norm(
            self.finger_tip_pos[0] - self.finger_tip_pos[2]) + np.linalg.norm(
            self.finger_tip_pos[0] - self.finger_tip_pos[3])) / 3
        self.grasp_dist = (np.linalg.norm(self.finger_tip_pos[0] - self.finger_tip_pos[1]) + np.linalg.norm(
            self.finger_tip_pos[0] - self.finger_tip_pos[2]) + np.linalg.norm(
            self.finger_tip_pos[0] - self.finger_tip_pos[3])) / 3

    def get_oracle_state(self):
        return np.concatenate([
            self.robot_qpos_vec, self.palm_v, self.palm_w, self.palm_pose.p, self.palm_vector[-1:],
            [float(self.current_step) / float(self.horizon)]
        ])

    def get_robot_state(self):
        return np.concatenate([
            self.robot_qpos_vec, self.palm_v, self.palm_w, self.palm_pose.p, self.palm_vector[-1:],
            [float(self.current_step) / float(self.horizon)]
        ])

    def get_reward(self, action):
        reward = 0
        reward += 0.2 * self.palm_vector[2]
        reward -= 0.2 * self.finger_base_touched_percent  # under no circumstances should hand touch bucket base
        if self.state == 1:
            reward = -0.1 * min(np.linalg.norm(self.palm_pose.p - self.handle_pose.p),
                                0.5)  # encourage palm be close to handle
        elif self.state == 2:
            reward += 0.2 * (int(self.is_contact))
            reward -= 0.1 * (int(self.is_arm_contact))
            reward -= 0.01 * np.linalg.norm(self.instance_base_link.get_velocity())
            reward -= 0.01 * np.linalg.norm(self.instance_base_link.get_angular_velocity())
            reward += 0.5 * self.progress
            reward += (0.2 * self.progress) * self.finger_touched_percent
        elif self.state == 3:
            reward += 0.2 * (int(self.is_contact))
            reward -= 0.1 * (int(self.is_arm_contact))
            reward += 0.5 * self.progress
            reward += (0.2 * self.progress) * self.finger_touched_percent
            if self.delta_height < 0.3:
                reward += 100 * (self.palm_height - self.last_palm_height)
                reward += self.delta_height / 0.3 * 10  # lift to 0.6m is enough

        action_penalty = np.sum(np.clip(self.robot.get_qvel(), -1, 1) ** 2) * 0.01
        controller_penalty = (self.cartesian_error ** 2) * 1e3
        reward -= 0.1 * (action_penalty + controller_penalty)
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
            raise NotImplementedError
        self.pos = self.instance_init_pos
        pos = self.pos + np.random.random(3) * self.rand_pos  # can add noise here to randomize loaded position
        random_orn = (np.random.rand() * 2 - 1) * self.rand_orn
        orn = transforms3d.euler.euler2quat(0, 0, np.pi + random_orn)
        self.instance.set_root_pose(sapien.Pose(pos, orn))
        # ================== update the init position state ==================
        self.init_height = self.instance_base_link.get_pose().p[2]
        self.init_pose_mat = transforms3d.quaternions.quat2mat(self.instance.get_root_pose().q)
        # ====================================================================
        self.update_cached_state()
        self.update_imagination(reset_goal=False)
        return self.get_observation()

    def setup_robot_annotation(self, robot_name: str):
        # here we load robot2bucket
        current_dir = Path(__file__).parent
        self.pos_path = current_dir.parent.parent.parent / "assets" / 'annotation' / f"bucket_{robot_name}_relative_position.json"
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
        # return 250
        return 100