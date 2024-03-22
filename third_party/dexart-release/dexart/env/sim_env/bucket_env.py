import os.path
from pathlib import Path

import numpy as np
import sapien.core as sapien
import transforms3d

from dexart.env.sim_env.base import BaseSimulationEnv
from dexart.env.task_setting import TASK_CONFIG
from termcolor import cprint
import json


class BucketEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, friction=0, iter=0, handle_type='left',
                 fix_root_link=False, **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **renderer_kwargs)
        self.instance_collision_links = None
        self.instance_links = None
        self.handle_link = None
        assert handle_type in ['left', 'middle']

        self.handle_type = handle_type
        self.fix_root_link = fix_root_link
        self.scale_path = None
        self.iter = iter
        # Construct scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(config=scene_config)
        self.scene.set_timestep(0.004)
        self.instance = None

        self.base_density = renderer_kwargs['density'] if renderer_kwargs.__contains__('density') else 7800

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
        self.task_config_name = 'bucket'
        self.instance_list = TASK_CONFIG[self.task_config_name]

        if isinstance(index, list):
            self.instance_list = index
            index = -1
        self.change_instance_when_reset = True if index == -1 else False
        cprint(f"[BucketEnv] change_instance_when_reset={self.change_instance_when_reset}", 'yellow')

        self.setup_instance_annotation()

        self.i = 0
        self.handle2link_relative_pose_dict = dict()

        if not self.change_instance_when_reset:
            self.index = self.instance_list[index]
            self.instance, self.revolute_joint, self.revolute_joint_index = self.load_instance(index=self.index)
            self.instance.set_qpos(self.joint_limits_dict[str(self.index)]['middle'])
            self.handle_link = self.revolute_joint.get_child_link()
            self.instance_links = self.instance.get_links()
            self.instance_collision_links = [link for link in self.instance.get_links() if
                                           len(link.get_collision_shapes()) > 0]
            self.instance_base_link = [link for link in self.instance.get_links() if link.get_name() == 'link_1'][0]
            self.handle_id = self.handle_link.get_id()
            self.instance_ids_without_handle = [link.get_id() for link in self.instance_links]
            self.instance_ids_without_handle.remove(self.handle_id)
            if not self.handle2link_relative_pose_dict.__contains__(self.index):
                self.handle2link_relative_pose_dict[self.index] = self.update_handle_relative_pose()
        self.reset_env()

    def setup_instance_annotation(self):
        current_dir = Path(__file__).parent
        self.scale_path = current_dir.parent.parent.parent / "assets" / "annotation" / "bucket_scale.json"
        with open(self.scale_path, "r") as f:
            self.scale_dict = json.load(f)
        self.joint_dicts = dict()
        for bucket_index in TASK_CONFIG['bucket']:
            joint_json_path = current_dir.parent.parent.parent / "assets" / "sapien" / (str(
                bucket_index) + "_thick_handle") / "mobility_v2.json"
            with open(joint_json_path, 'r') as load_f:
                load_dict = json.load(load_f)
            self.joint_dicts[bucket_index] = load_dict

        self.joint_limits_dict_path = current_dir.parent.parent.parent / "assets" / "annotation" / "bucket_joint_annotation.json"
        with open(self.joint_limits_dict_path, "r") as f:
            self.joint_limits_dict = json.load(f)

    def load_instance(self, index):
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.load_multiple_collisions_from_file = True
        loader.fix_root_link = self.fix_root_link
        current_dir = Path(__file__).parent
        urdf_path = str(
            current_dir.parent.parent.parent / "assets" / "sapien" / (str(index) + "_thick_handle") / "mobility.urdf")


        loader.scale = self.scale_dict[str(index)]
        physical_material: sapien.PhysicalMaterial = self.scene.create_physical_material(
            static_friction=1,
            dynamic_friction=1,
            restitution=0.1,
        )
        instance: sapien.Articulation = loader.load(urdf_path, config={'material': physical_material,
                                                                       'link': {'link_0': {'density': 100},
                                                                                'link_1':  {'density': self.base_density}}})
        for joint in instance.get_joints():
            joint.set_friction(self.friction)
            # joint.set_drive_property(0, 5)

        load_dict = self.joint_dicts[index]
        joint_size = len(load_dict)

        # in sapien, it will auto add the fixed base joint, so the loaded joint_size need to plus 1
        # assert len(instance.get_joints()) == joint_size + 1
        dof = 0
        revolute_joint = None
        for i, joint_entry in enumerate(load_dict):
            if joint_entry['joint'] == 'free':  # except static type
                dof += 1
            if joint_entry['joint'] == 'hinge' and joint_entry['name'] == 'handle':
                revolute_joint_index = dof - 1
                revolute_joint = instance.get_active_joints()[revolute_joint_index]
        assert dof == instance.dof, "dof parse error, index={}, calculate_dof={}, real_dof={}".format(index, dof,
                                                                                                    instance.dof)
        assert revolute_joint, "revolue_joint can not be None!"



        if self.joint_limits_dict[str(index)].__contains__('min'):
            q_limits = instance.get_qlimits()
            q_limits[revolute_joint_index][0] = min(self.joint_limits_dict[str(index)]['min'], self.joint_limits_dict[str(index)]['max'])
            q_limits[revolute_joint_index][1] = max(self.joint_limits_dict[str(index)]['min'], self.joint_limits_dict[str(index)]['max'])
            for k, joint in enumerate(instance.get_active_joints()):
                joint.set_limits(q_limits[k:k + 1])


        return instance, revolute_joint, revolute_joint_index

    def reset_env(self):
        if self.change_instance_when_reset:
            if self.instance is not None:
                self.scene.remove_articulation(self.instance)
                self.instance, self.revolute_joint, self.revolute_joint_index = None, None, None
            self.i = (self.i + 1) % len(self.instance_list)
            self.index = self.instance_list[self.i]
            self.instance, self.revolute_joint, self.revolute_joint_index = self.load_instance(index=self.index)
            self.instance.set_qpos(self.joint_limits_dict[str(self.index)]['middle'])
            self.handle_link = self.revolute_joint.get_child_link()
            self.instance_links = self.instance.get_links()
            self.instance_collision_links = [link for link in self.instance.get_links() if
                                           len(link.get_collision_shapes()) > 0]
            self.instance_base_link = [link for link in self.instance.get_links() if link.get_name() == 'link_1'][0]
            self.handle_id = self.handle_link.get_id()
            self.instance_ids_without_handle = [link.get_id() for link in self.instance_links]
            self.instance_ids_without_handle.remove(self.handle_id)
            if not self.handle2link_relative_pose_dict.__contains__(self.index):
                self.handle2link_relative_pose_dict[self.index] = self.update_handle_relative_pose()

        pos = self.pos  # can add noise here to randomize loaded position
        orn = transforms3d.euler.euler2quat(0, 0, 0)

        self.instance.set_root_pose(sapien.Pose(pos, orn))
        self.instance.set_qpos(self.joint_limits_dict[str(self.index)]['left'])

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
            z = vertex_global_pose.p[2]
            if z > z_max:
                z_max = z
                max_z_index = i
        mean_pos = sum_pos / len(vertices_global_pose_list)

        # for x and z, we use the corresponding value of the highest vertex
        # for y, we use the mean of all the vertices
        x = vertices_global_pose_list[max_z_index].p[0]
        z = z_max
        y = mean_pos[1]

        handle_global_pose = sapien.Pose(np.array([x, y, z]))
        link_global_pose = self.handle_link.get_pose()

        relative_pose = link_global_pose.inv().transform(handle_global_pose)
        return relative_pose

    def get_handle_global_pose(self):
        better_global_pose = self.handle_link.get_pose().transform(self.handle2link_relative_pose_dict[self.index])
        return better_global_pose