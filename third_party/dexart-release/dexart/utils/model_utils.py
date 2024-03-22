import numpy as np
from sapien import core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat
from typing import List


def build_free_root(robot_builder: sapien.ArticulationBuilder, robot_name="", translation_range=(-1, 1),
                    rotation_range=(-np.pi, np.pi), rotate_final=True):
    builders = []
    parent = robot_builder.create_link_builder()
    builders.append(parent)
    joint_types = ["prismatic"] * 3 + ["revolute"] * 3
    joint_limit = [translation_range] * 3 + [rotation_range] * 3
    joint_name = [f"{name}_joint" for name in "xyz"] + [f"{name}_rotation_joint" for name in "xyz"]
    for i in range(6):
        parent.set_mass_and_inertia(1e-4, Pose(np.zeros(3)), np.ones(3) * 1e-6)
        parent.set_name(f'root{i}_{robot_name}')
        child = robot_builder.create_link_builder(parent)
        child.set_joint_name(joint_name[i])
        if i == 3 or i == 0:
            child.set_joint_properties(joint_types[i], limits=np.array([joint_limit[i]]))
        elif i == 4 or i == 1:
            child.set_joint_properties(joint_types[i], limits=np.array([joint_limit[i]]),
                                       pose_in_child=Pose(q=euler2quat(0, 0, np.pi / 2)),
                                       pose_in_parent=Pose(q=euler2quat(0, 0, np.pi / 2)))
        elif i == 2 or (i == 5 and not rotate_final):
            child.set_joint_properties(joint_types[i], limits=np.array([joint_limit[i]]),
                                       pose_in_parent=Pose(q=euler2quat(0, -np.pi / 2, 0)),
                                       pose_in_child=Pose(q=euler2quat(0, -np.pi / 2, 0)))
        elif i == 5 and rotate_final:
            child.set_joint_properties(joint_types[i], limits=np.array([joint_limit[i]]),
                                       pose_in_parent=Pose(q=euler2quat(0, -np.pi / 2, 0)),
                                       pose_in_child=Pose(q=euler2quat(-np.pi / 2, np.pi, 0, "sxyz")))
        parent = child
        builders.append(parent)

    return parent


def build_ball_joint(robot_builder: sapien.ArticulationBuilder, parent: sapien.LinkBuilder, name: str,
                     pose: Pose) -> List[sapien.ArticulationBuilder]:
    link1 = robot_builder.create_link_builder(parent)
    link2 = robot_builder.create_link_builder(link1)
    link3 = robot_builder.create_link_builder(link2)
    links = [link1, link2, link3]
    joint_limit = np.array([[-np.pi, np.pi]])
    pose_in_parent = [pose, Pose(q=euler2quat(0, 0, np.pi / 2)), Pose(q=euler2quat(0, -np.pi / 2, 0))]
    pose_in_child = [Pose(), Pose(q=euler2quat(0, 0, np.pi / 2)), Pose(q=euler2quat(0, -np.pi / 2, 0))]
    for i in range(3):
        link = links[i]
        link.set_name(f"{name}_{i}_link")
        link.set_joint_name(f"{name}_{i}_joint")
        link.set_joint_properties("revolute", limits=joint_limit, pose_in_parent=pose_in_parent[i],
                                  pose_in_child=pose_in_child[i])
    return links


def rot_from_connected_link(pos_parent, pos_child, parent_rotation=np.eye(3), up_axis=np.array([0, 1, 0])):
    x_axis = parent_rotation.T @ (pos_parent - pos_child)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = up_axis - np.sum(up_axis * x_axis) * x_axis
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    mat = np.stack([x_axis, y_axis, z_axis], axis=1)
    return mat


def fix_link_inertia(robot_builder: sapien.ArticulationBuilder):
    for builder in robot_builder.get_link_builders():
        if len(builder.get_collisions()) < 1:
            builder.set_mass_and_inertia(1e-2, Pose(np.zeros(3)), np.ones(3) * 1e-4)


def create_visual_material(renderer: sapien.VulkanRenderer, specular, metallic, roughness, base_color):
    if renderer is None:
        return None
    viz_mat = renderer.create_material()
    viz_mat.set_specular(specular)
    viz_mat.set_metallic(metallic)
    viz_mat.set_roughness(roughness)
    viz_mat.set_base_color(np.array(base_color))
    return viz_mat
