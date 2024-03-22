from pathlib import Path
import numpy as np
import sapien.core as sapien
import transforms3d
from dexart.env.sim_env.base import BaseSimulationEnv
from dexart.env.task_setting import TASK_CONFIG
from termcolor import cprint
import json


class FaucetEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, friction=5, iter=0, **renderer_kwargs):
        super().__init__(use_gui=use_gui, frame_skip=frame_skip, **renderer_kwargs)

        self.rand_pos = renderer_kwargs.get('rand_pos', 0.0)

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
        self.task_config_name = 'faucet'
        self.instance_list = TASK_CONFIG[self.task_config_name]
        if isinstance(index, list):
            self.instance_list = index
            index = -1
        self.change_instance_when_reset = True if index == -1 else False
        cprint(f"[FaucetEnv] change_instance_when_reset={self.change_instance_when_reset}", 'yellow')
        self.setup_instance_annotation()
        self.i = 0

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
        self.reset_env()

    def setup_instance_annotation(self):
        current_dir = Path(__file__).parent
        self.scale_path = current_dir.parent.parent.parent / "assets" / "annotation" / "faucet_scale.json"
        with open(self.scale_path, "r") as f:
            self.scale_dict = json.load(f)
        self.joint_dicts = dict()
        for instance_index in TASK_CONFIG['faucet']:
            joint_json_path = current_dir.parent.parent.parent / "assets" / "sapien" / str(
                instance_index) / "mobility_v2.json"
            with open(joint_json_path, 'r') as load_f:
                load_dict = json.load(load_f)
            self.joint_dicts[instance_index] = load_dict

    def load_instance(self, index):
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.load_multiple_collisions_from_file = True
        current_dir = Path(__file__).parent
        urdf_path = str(current_dir.parent.parent.parent / "assets" / "sapien" / str(index) / "mobility.urdf")
        loader.scale = self.scale_dict[str(index)]

        instance: sapien.Articulation = loader.load(urdf_path, config={'density': 7800})
        for joint in instance.get_joints():
            joint.set_friction(5)
            # joint.set_drive_property(0, 5)

        load_dict = self.joint_dicts[index]
        joint_size = len(load_dict)
        # in sapien, it will auto add the fixed base joint, so the loaded joint_size need to plus 1
        assert len(instance.get_joints()) == joint_size + 1
        dof = 0
        revolute_joint = None
        for i, joint_entry in enumerate(load_dict):
            if joint_entry['joint'] == 'hinge' or joint_entry['joint'] == 'slider':  # except static type
                dof += 1
            if joint_entry['joint'] == 'hinge' and joint_entry['name'] == 'switch':
                revolute_joint_index = dof - 1
                revolute_joint = instance.get_active_joints()[revolute_joint_index]
        assert dof == instance.dof, "dof parse error, index={}, calculate_dof={}, real_dof={}".format(index, dof,
                                                                                                      instance.dof)
        assert revolute_joint, "revolue_joint can not be None!"
        return instance, revolute_joint, revolute_joint_index

    def reset_env(self):
        if self.change_instance_when_reset:
            if self.instance is not None:
                self.scene.remove_articulation(self.instance)
                self.instance, self.revolute_joint, self.revolute_joint_index = None, None, None
            # self.index = self.instance_list[random.randint(0, len(self.instance_list) - 1)]
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
        pos = self.pos  # can add noise here to randomize loaded position
        orn = transforms3d.euler.euler2quat(0, 0, 0)
        self.instance.set_root_pose(sapien.Pose(pos, orn))
        self.instance.set_qpos(np.zeros(self.instance.dof))