from typing import List

import numpy as np
import sapien.core as sapien


class PartialKinematicModel:
    def __init__(self, robot: sapien.Articulation, start_joint_name: str, end_joint_name: str):
        self.original_robot = robot
        self.start_joint_tuple = \
            [(joint, num) for num, joint in enumerate(robot.get_joints()) if
             joint.get_name() == start_joint_name][0]
        self.end_joint_tuple = \
            [(joint, num) for num, joint in enumerate(robot.get_joints()) if joint.get_name() == end_joint_name][
                0]
        self.start_link = self.start_joint_tuple[0].get_parent_link()
        self.end_link = self.end_joint_tuple[0].get_child_link()

        # Build new articulation for partial kinematics chain
        scene = robot.get_builder().get_scene()
        builder = scene.create_articulation_builder()
        root = builder.create_link_builder()
        root.set_mass_and_inertia(
            self.start_link.get_mass(),
            self.start_link.cmass_local_pose,
            self.start_link.get_inertia(),
        )
        links = [root]
        all_joints = robot.get_joints()[self.start_joint_tuple[1]: self.end_joint_tuple[1] + 1]
        for j_idx, j in enumerate(all_joints):
            link = builder.create_link_builder(links[-1])
            link.set_mass_and_inertia(
                j.get_child_link().get_mass(),
                j.get_child_link().cmass_local_pose,
                j.get_child_link().get_inertia(),
            )
            link.set_joint_properties(
                j.type, j.get_limits(), j.get_pose_in_parent(), j.get_pose_in_child()
            )
            link.set_name(j.get_child_link().get_name())
            links.append(link)

        partial_robot = builder.build(fix_root_link=True)
        partial_robot.set_pose(sapien.Pose([0, 0, -10]))
        self.model = partial_robot.create_pinocchio_model()

        # Parse new model
        self.dof = partial_robot.dof
        self.end_link_name = self.end_link.get_name()
        self.end_link_index = [i for i, link in enumerate(partial_robot.get_links()) if
                               link.get_name() == self.end_link_name][0]
        self.partial_robot = partial_robot

    def compute_end_link_spatial_jacobian(self, partial_qpos):
        self.partial_robot.set_qpos(partial_qpos)
        jacobian = self.partial_robot.compute_world_cartesian_jacobian()[
                   self.end_link_index * 6 - 6: self.end_link_index * 6, :]
        return jacobian


class SAPIENKinematicsModelStandalone:
    def __init__(self, urdf_path):
        self.engine = sapien.Engine()
        self.scene = self.engine.create_scene()
        loader = self.scene.create_urdf_loader()
        self.robot = loader.load(urdf_path)
        self.scene.step()
        self.robot.set_pose(sapien.Pose())
        self.robot_model = self.robot.create_pinocchio_model()
        self.joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        self.link_name2id = {self.robot.get_links()[i].get_name(): i for i in range(len(self.robot.get_links()))}

        self.cached_mapping = []
        self.cached_names = ""

    def get_link_pose(self, qpos, joint_names: List[str], link_name: str):
        cat_names = "-".join(joint_names)
        if cat_names == self.cached_names:
            forward_mapping = self.cached_mapping
        else:
            print(f"Build new cached names")
            forward_mapping, _ = self.get_bidir_mapping(joint_names)
            self.cached_names = cat_names
            self.cached_mapping = forward_mapping

        inner_qpos = np.array(qpos)[forward_mapping]
        self.robot_model.compute_forward_kinematics(inner_qpos)

        link_index = self.link_name2id[link_name]
        pose = self.robot_model.get_link_pose(link_index)
        return np.concatenate([pose.p, pose.q])

    def get_bidir_mapping(self, joint_names: List[str]):
        assert len(joint_names) == len(self.joint_names)
        forward_mapping = []
        backward_mapping = []
        for joint_name in self.joint_names:
            index = joint_names.index(joint_name)
            forward_mapping.append(index)
        for joint_name in joint_names:
            index = self.joint_names.index(joint_name)
            backward_mapping.append(index)
        return forward_mapping, backward_mapping
