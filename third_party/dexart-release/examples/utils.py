#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from dexart.env.task_setting import ROBUSTNESS_INIT_CAMERA_CONFIG
import open3d as o3d

def visualize_observation(obs, use_seg=False, img_type=None):
    def visualize_pc_with_seg_label(cloud):
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud[:, :3]))

        def map(feature):
            color = np.zeros((feature.shape[0], 3))
            COLOR20 = np.array(
                [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                 [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
                 [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
                 [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]]) / 255
            for i in range(feature.shape[0]):
                for j in range(feature.shape[1]):
                    if feature[i, j] == 1:
                        color[i, :] = COLOR20[j, :]
            return color

        color = map(cloud[:, 3:])
        pc.colors = o3d.utility.Vector3dVector(color)
        return pc

    pc = obs["instance_1-point_cloud"]
    if use_seg:
        gt_seg = obs["instance_1-seg_gt"]
        pc = np.concatenate([pc, gt_seg], axis=1)
    pc = visualize_pc_with_seg_label(pc)
    if img_type == "robot":
        robot_pc = obs["imagination_robot"]
        pc += visualize_pc_with_seg_label(robot_pc)
    else:
        raise NotImplementedError
    return pc


def get_viewpoint_camera_parameter():
    robustness_init_camera_config = ROBUSTNESS_INIT_CAMERA_CONFIG['laptop']
    r = robustness_init_camera_config['r']
    phi = robustness_init_camera_config['phi']
    theta = robustness_init_camera_config['theta']
    center = robustness_init_camera_config['center']

    x0, y0, z0 = center
    # phi in [0, pi/2]
    # theta in [0, 2 * pi]
    x = x0 + r * np.sin(phi) * np.cos(theta)
    y = y0 + r * np.sin(phi) * np.sin(theta)
    z = z0 + r * np.cos(phi)

    cam_pos = np.array([x, y, z])
    forward = np.array([x0 - x, y0 - y, z0 - z])
    forward /= np.linalg.norm(forward)

    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)

    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos
    return cam_pos, center, up, mat44