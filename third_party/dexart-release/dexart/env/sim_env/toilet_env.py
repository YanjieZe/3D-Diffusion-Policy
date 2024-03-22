import os
import random
from pathlib import Path
import numpy as np
import sapien.core as sapien
import transforms3d
from dexart.env.sim_env.base import BaseSimulationEnv
from dexart.env.task_setting import TASK_CONFIG
from termcolor import cprint
import json


class ToiletEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, friction=5, iter=0, **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **renderer_kwargs)
        self.instance_collision_links = None
        self.instance_links = None
        self.handle_link = None
        self.handle2link_relative_pose = None
        self.scale_path = None
        self.iter = iter
        # Construct scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(config=scene_config)
        self.scene.set_timestep(0.004)
        self.instance = None

        # Dummy camera creation to initial geometry object
        if self.renderer and not self.no_rgb:
            cam = self.scene.add_camera("init_not_used", width=10, height=10, fovy=1, near=0.1, far=1)
            self.scene.remove_camera(cam)

        self.friction = friction
        # load table
        self.table = self.create_table(table_height=0.6, table_half_size=[0.65, 0.65, 0.025])
        self.create_room()
        # default pos and orn, will be used in reset_env
        self.pos = np.array([0, 0, 0.1])
        self.orn = transforms3d.euler.euler2quat(0, 0, 0)

        index = renderer_kwargs['index']
        self.task_config_name = 'toilet'
        self.instance_list = TASK_CONFIG[self.task_config_name]
        if isinstance(index, list):
            self.instance_list = index
            index = -1
        self.change_instance_when_reset = True if index == -1 else False
        cprint(f"[ToiletEnv] change_instance_when_reset={self.change_instance_when_reset}", 'yellow')

        self.i = 0
        self.handle2link_relative_pose_dict = dict()
        self.setup_instance_annotation()
        # for toilet env
        self.init_open_rad = 0.25

        if not self.change_instance_when_reset:
            self.index = self.instance_list[index]
            self.instance, self.revolute_joint, self.revolute_joint_index = self.load_instance(index=self.index)
            self.handle_link = self.revolute_joint.get_child_link()
            self.instance_links = self.instance.get_links()
            self.instance_collision_links = [link for link in self.instance.get_links() if
                                             len(link.get_collision_shapes()) > 0]
            self.handle_id = self.handle_link.get_id()
            self.instance_ids_without_handle = [link.get_id() for link in self.instance_links]
            self.instance_ids_without_handle.remove(self.handle_id)
            if not self.handle2link_relative_pose_dict.__contains__(self.index):
                self.handle2link_relative_pose_dict[self.index] = self.update_handle_relative_pose()
        self.reset_env()

    def setup_instance_annotation(self):
        current_dir = Path(__file__).parent
        self.scale_path = current_dir.parent.parent.parent / "assets" / "annotation" / "toilet_scale.json"
        if os.path.exists(self.scale_path):
            with open(self.scale_path, "r") as f:
                self.scale_dict = json.load(f)
        else:
            raise FileNotFoundError
        self.joint_dicts = dict()
        self.joint_limits_dict_path = current_dir.parent.parent.parent / "assets" / "annotation" / "toilet_joint_annotation.json"
        self.joint_limits_dict = dict()
        if os.path.exists(self.joint_limits_dict_path):
            with open(self.joint_limits_dict_path, "r") as f:
                self.joint_limits_dict = json.load(f)
        self.joint_info_path = current_dir.parent.parent.parent / "assets" / "annotation" / "toilet_joint.txt"
        self.joint_info = dict()
        with open(self.joint_info_path) as f:
            lines = f.readlines()
            for line in lines[1:]:
                info_list = line.strip().split(" ")
                index = info_list[0]
                lid_joint_name = info_list[1]
                lid_link_name = info_list[2]
                body_link_name = info_list[3]
                seat_link_name = None if len(info_list) <= 4 else info_list[4]
                self.joint_info[int(index)] = dict(lid_joint_name=lid_joint_name, lid_link_name=lid_link_name,
                                              body_link_name=body_link_name, seat_link_name=seat_link_name)


    def load_instance(self, index):
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.load_multiple_collisions_from_file = True
        loader.fix_root_link = True
        current_dir = Path(__file__).parent
        urdf_path = str(current_dir.parent.parent.parent / "assets"  / "sapien" / str(index) / "mobility.urdf")
        loader.scale = self.scale_dict[str(index)] if self.scale_dict.__contains__(str(index)) else 1

        instance: sapien.Articulation = loader.load(urdf_path, config={'density': 50})
        for joint in instance.get_joints():
            joint.set_friction(self.friction)

        for i, joint in enumerate(instance.get_active_joints()):
            if joint.get_name() == self.joint_info[index]["lid_joint_name"]:
                revolute_joint_index = i
                revolute_joint = instance.get_active_joints()[revolute_joint_index]
                break
        assert revolute_joint, "revolue_joint can not be None!" + str(index)

        self.lid_link = [link for link in instance.get_links() if link.get_name() == self.joint_info[index]["lid_link_name"]][0]
        if self.joint_info[index]["seat_link_name"] is not None:
            self.seat_link = [link for link in instance.get_links() if link.get_name() == self.joint_info[index]["seat_link_name"]][0]
            for collision_shape in self.lid_link.get_collision_shapes():
                group0, group1, group2, group3 = collision_shape.get_collision_groups()
                collision_shape.set_collision_groups(group0, group1, 1, group3)
            for collision_shape in self.seat_link.get_collision_shapes():
                group0, group1, group2, group3 = collision_shape.get_collision_groups()
                collision_shape.set_collision_groups(group0, group1, 1, group3)

        q_limits = instance.get_qlimits()
        for i in range(instance.dof):
            if i == revolute_joint_index:
                continue
            else:
                q_limits[i][1] = q_limits[i][0]
        for k, joint in enumerate(instance.get_active_joints()):
            joint.set_limits(q_limits[k:k+1])
        q = np.zeros(instance.dof)
        q[revolute_joint_index] = self.joint_limits_dict[str(index)]['left'] + self.init_open_rad
        return instance, revolute_joint, revolute_joint_index

    def reset_env(self):
        if self.change_instance_when_reset:
            if self.instance is not None:
                self.scene.remove_articulation(self.instance)
                self.instance, self.revolute_joint, self.revolute_joint_index = None, None, None

            self.i = (self.i + 1) % len(self.instance_list)
            self.index = self.instance_list[self.i]
            self.instance, self.revolute_joint, self.revolute_joint_index = self.load_instance(index=self.index)
            self.handle_link = self.revolute_joint.get_child_link()
            self.instance_links = self.instance.get_links()
            self.instance_collision_links = [link for link in self.instance.get_links() if
                                             len(link.get_collision_shapes()) > 0]
            self.handle_id = self.handle_link.get_id()
            self.instance_ids_without_handle = [link.get_id() for link in self.instance_links]
            self.instance_ids_without_handle.remove(self.handle_id)
            if not self.handle2link_relative_pose_dict.__contains__(self.index):
                self.handle2link_relative_pose_dict[self.index] = self.update_handle_relative_pose()
        pos = self.pos  # can add noise here to randomize loaded position
        orn = transforms3d.euler.euler2quat(0, 0, 0)
        self.instance.set_root_pose(sapien.Pose(pos, orn))
        q = np.zeros(self.instance.dof)
        q[self.revolute_joint_index] = self.joint_limits_dict[str(self.index)]['left'] + self.init_open_rad
        self.instance.set_qpos(q)

    def update_handle_relative_pose(self):
        vertices_relative_pose_list = list()
        vertices_global_pose_list = list()
        # get all the collision mesh of laptop upper face
        for collision_mesh in self.handle_link.get_collision_shapes():
            vertices = collision_mesh.geometry.vertices
            for vertex in vertices:
                vertex_relative_pose = sapien.Pose(vertex * collision_mesh.geometry.scale).transform(
                    collision_mesh.get_local_pose())
                vertices_relative_pose_list.append(vertex_relative_pose)
                vertices_global_pose_list.append(self.handle_link.get_pose().transform(vertex_relative_pose))

        z_max = -1e9
        max_z_index = 0
        sum_pos = np.zeros(3)
        for i, vertex_global_pose in enumerate(vertices_global_pose_list):
            sum_pos += vertex_global_pose.p
            z = vertex_global_pose.p[0]
            if z > z_max:
                z_max = z
                max_z_index = i
        mean_pos = sum_pos / len(vertices_global_pose_list)

        # for x and z, we use the corresponding value of the highest vertex
        # for y, we use the mean of all the vertices
        x = vertices_global_pose_list[max_z_index].p[0]
        y = vertices_global_pose_list[max_z_index].p[1]
        y = mean_pos[1]
        z = vertices_global_pose_list[max_z_index].p[2]

        handle_global_pose = sapien.Pose(np.array([x, y, z]))
        link_global_pose = self.handle_link.get_pose()

        relative_pose = link_global_pose.inv().transform(handle_global_pose)
        return relative_pose

    def get_handle_global_pose(self):
        better_global_pose = self.handle_link.get_pose().transform(self.handle2link_relative_pose_dict[self.index])
        return better_global_pose