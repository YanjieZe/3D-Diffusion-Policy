from functools import cached_property
from pathlib import Path
from typing import Optional, List, Dict, Sequence, Union

import numpy as np
import sapien.core as sapien
import transforms3d.quaternions
from sapien.core import Pose
from sapien.utils import Viewer

from dexart.env.sim_env.constructor import get_engine_and_renderer, add_default_scene_light
from dexart.utils.random_utils import np_random


def recover_action(action, limit):
    action = (action + 1) / 2 * (limit[:, 1] - limit[:, 0]) + limit[:, 0]
    return action


class BaseSimulationEnv(object):
    def __init__(self, use_gui=False, frame_skip=5, use_visual_obs=False, no_rgb=False, need_offscreen_render=True,
                 **renderer_kwargs):
        need_offscreen_render = need_offscreen_render or use_visual_obs

        engine, renderer = get_engine_and_renderer(use_gui=use_gui, need_offscreen_render=need_offscreen_render,
                                                   no_rgb=no_rgb, **renderer_kwargs)
        self.use_gui = use_gui
        self.engine = engine
        self.renderer = renderer
        self.frame_skip = frame_skip

        self.np_random = None
        self.viewer: Optional[Viewer] = None
        self.scene: Optional[sapien.Scene] = None
        self.robot: Optional[sapien.Articulation] = None
        self.init_state: Optional[Dict] = None
        self.robot_name = ""

        # Camera
        self.use_visual_obs = use_visual_obs
        self.use_offscreen_render = need_offscreen_render
        self.no_rgb = no_rgb and not use_gui
        self.cameras: Dict[str, sapien.CameraEntity] = {}

        self.seed()
        self.current_step = 0

    def simple_step(self):
        self.pre_step()
        for i in range(self.frame_skip):
            self.scene.step()
        self.post_step()
        self.current_step += 1

    def pre_step(self):
        pass

    def post_step(self):
        pass

    def reset_env(self):
        raise NotImplementedError

    def __del__(self):
        self.scene = None

    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        return [seed]

    def set_seed(self, seed=None):
        self.seed(seed)

    def render(self, mode="human"):
        assert self.use_gui
        if mode == 'human':
            if self.viewer is None:
                self.viewer = Viewer(self.renderer)
                self.viewer.set_scene(self.scene)
            if len(self.scene.get_all_lights()) <= 1:
                add_default_scene_light(self.scene, self.renderer)
            self.viewer.render()
            return self.viewer
        else:
            raise NotImplementedError

    def check_contact(self, actors1: List[sapien.Actor], actors2: List[sapien.Actor], impulse_threshold=1e-2) -> bool:
        actor_set1 = set(actors1)
        actor_set2 = set(actors2)
        for contact in self.scene.get_contacts():
            contact_actors = {contact.actor0, contact.actor1}
            if len(actor_set1 & contact_actors) > 0 and len(actor_set2 & contact_actors) > 0:
                impulse = [point.impulse for point in contact.points]
                if np.sum(np.abs(impulse)) < impulse_threshold:
                    continue
                return True
        return False

    def check_actor_pair_contact(self, actor1: sapien.Actor, actor2: sapien.Actor, impulse_threshold=1e-2) -> bool:
        actor_pair = {actor1, actor2}
        for contact in self.scene.get_contacts():
            contact_actors = {contact.actor0, contact.actor1}
            if contact_actors == actor_pair:
                impulse = [point.impulse for point in contact.points]
                if np.sum(np.abs(impulse)) < impulse_threshold:
                    continue
                return True
        return False

    def check_actor_pair_contacts(self, actors1: List[sapien.Actor], actor2: sapien.Actor,
                                  impulse_threshold=1e-2) -> np.ndarray:
        actor_set1 = set(actors1)
        contact_buffer = np.zeros(len(actors1))
        for contact in self.scene.get_contacts():
            contact_actors = {contact.actor0, contact.actor1}
            # print("contact_acotrs = ", contact_actors)
            if len(actor_set1 & contact_actors) > 0 and actor2 in contact_actors:
                impulse = [point.impulse for point in contact.points]
                if np.sum(np.abs(impulse)) < impulse_threshold:
                    continue
                contact_actors.remove(actor2)
                actor_index = actors1.index(contact_actors.pop())
                contact_buffer[actor_index] = 1
        return contact_buffer

    def check_actors_pair_contacts(self, actors1: List[sapien.Actor], actors2: List[sapien.Actor],
                                   impulse_threshold=1e-2) -> np.ndarray:
        contact_buffer = np.zeros(len(actors1))
        for actor2 in actors2:
            contact_buffer_local = self.check_actor_pair_contacts(actors1, actor2, impulse_threshold)
            contact_buffer += contact_buffer_local
        contact_buffer = np.clip(contact_buffer, 0, 1)
        return contact_buffer

    def check_actor_pair_contacts_in_distances(self, actors1: List[sapien.Actor], actor2: sapien.Actor,
                                               centers: List[np.ndarray], radii: List[float],
                                              impulse_threshold=1e-2, reverse=False) -> np.ndarray:
        actor_set1 = set(actors1)
        contact_buffer = np.zeros(len(actors1))
        for contact in self.scene.get_contacts():
            contact_actors = {contact.actor0, contact.actor1}
            # print("contact_acotrs = ", contact_actors)
            if len(actor_set1 & contact_actors) > 0 and actor2 in contact_actors:
                impulse = [point.impulse for point in contact.points]

                distance = np.ones(len(contact.points))
                for center, radius in zip(centers, radii):
                    if reverse:
                        distance = np.logical_and(distance, np.array([np.linalg.norm(point.position - center) < radius for point in contact.points]))
                    else:
                        distance = np.logical_and(distance, np.array([np.linalg.norm(point.position - center) > radius for point in contact.points]))
                if (np.sum(np.abs(impulse)) < impulse_threshold) or np.sum(np.array(distance)) > 0:
                    continue
                contact_actors.remove(actor2)
                actor_index = actors1.index(contact_actors.pop())
                contact_buffer[actor_index] = 1
        return contact_buffer

    def check_actors_pair_contacts_in_distance(self, actors1: List[sapien.Actor], actors2: List[sapien.Actor],
                                               centers: List[np.ndarray], radii: List[float],
                                               impulse_threshold=1e-2, reverse=False) -> np.ndarray:
        contact_buffer = np.zeros(len(actors1))
        for actor2 in actors2:
            contact_buffer_local = self.check_actor_pair_contacts_in_distances(actors1, actor2, centers, radii,
                                                                               impulse_threshold, reverse=reverse)
            contact_buffer += contact_buffer_local
        contact_buffer = np.clip(contact_buffer, 0, 1)
        return contact_buffer


    @cached_property
    def joint_limits(self):
        return self.robot.get_qlimits()

    def create_viewer(self):
        viewer = Viewer(renderer=self.renderer)
        viewer.set_scene(self.scene)
        viewer.set_camera_xyz(x=-4, y=0, z=2)
        # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
        # The camera now looks at the origin
        viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
        return viewer

    def create_table(self, table_height=1.0, table_half_size=(0.8, 0.8, 0.025)):
        builder = self.scene.create_actor_builder()

        # Top
        top_pose = sapien.Pose([0, 0, -table_half_size[2]])
        top_material = self.scene.create_physical_material(1, 0.5, 0.01)
        builder.add_box_collision(pose=top_pose, half_size=table_half_size, material=top_material)
        # Leg
        asset_dir = Path(__file__).parent.parent.parent.parent / "assets"
        table_map_path = asset_dir / "misc" / "table_map.jpg"
        table_cube_path = asset_dir / "misc" / "cube.obj"
        if self.renderer and not self.no_rgb:
            table_visual_material = self.renderer.create_material()
            table_visual_material.set_metallic(0.0)
            table_visual_material.set_specular(0.3)
            table_visual_material.set_diffuse_texture_from_file(str(table_map_path))
            # table_visual_material.set_base_color(np.array([1, 1, 1, 1]))
            table_visual_material.set_roughness(0.3)
            leg_size = np.array([0.025, 0.025, (table_height / 2 - table_half_size[2])])
            leg_height = -table_height / 2 - table_half_size[2]
            x = table_half_size[0] - 0.1
            y = table_half_size[1] - 0.1
            if self.use_gui or self.use_visual_obs:
                builder.add_visual_from_file(str(table_cube_path), pose=top_pose, material=table_visual_material,
                                             scale=table_half_size, name="surface")
                builder.add_box_visual(pose=sapien.Pose([x, y, leg_height]), half_size=leg_size,
                                       material=table_visual_material, name="leg0")
                builder.add_box_visual(pose=sapien.Pose([x, -y, leg_height]), half_size=leg_size,
                                       material=table_visual_material, name="leg1")
                builder.add_box_visual(pose=sapien.Pose([-x, y, leg_height]), half_size=leg_size,
                                       material=table_visual_material, name="leg2")
                builder.add_box_visual(pose=sapien.Pose([-x, -y, leg_height]), half_size=leg_size,
                                       material=table_visual_material, name="leg3")
        return builder.build_static("table")

    def create_box(self,
                   pose: sapien.Pose,
                   half_size,
                   color=None,
                   name='',
                   ) -> sapien.Actor:
        """Create a box.

        Args:
            scene: sapien.Scene to create a box.
            pose: 6D pose of the box.
            half_size: [3], half size along x, y, z axes.
            color: [3] or [4], rgb or rgba
            name: name of the actor.

        Returns:
            sapien.Actor
        """
        half_size = np.array(half_size)
        builder: sapien.ActorBuilder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)  # Add collision shape
        builder.add_box_visual(half_size=half_size, color=color)  # Add visual shape
        box: sapien.Actor = builder.build(name=name)
        # Or you can set_name after building the actor
        # box.set_name(name)
        box.set_pose(pose)
        return box

    def create_room(self, length=3):
        room_color = [20.0] * 3
        box = self.create_box(
            sapien.Pose(p=np.array([-length, 0., 0])),
            half_size=np.array([1, 100, 100]),
            color=room_color,
            name='box1',
        )
        box.lock_motion()
        box = self.create_box(
            sapien.Pose(p=np.array([length, 0., 0])),
            half_size=np.array([1, 100, 100]),
            color=room_color,
            name='box1',
        )
        box.lock_motion()
        box = self.create_box(
            sapien.Pose(p=np.array([0, length, 0])),
            half_size=np.array([100, 1, 100]),
            color=room_color,
            name='box1',
        )
        box.lock_motion()
        box = self.create_box(
            sapien.Pose(p=np.array([0, -length, 0])),
            half_size=np.array([100, 1, 100]),
            color=room_color,
            name='box1',
        )
        box.lock_motion()

        box = self.create_box(
            sapien.Pose(p=np.array([0, 0, 9])),
            half_size=np.array([100, 100, 1]),
            color=room_color,
            name='box1',
        )
        box.lock_motion()

    def create_camera(self, position: np.ndarray, look_at_dir: np.ndarray, right_dir: np.ndarray, name: str,
                      resolution: Sequence[Union[float, int]], fov: float, mount_actor_name: str = None):
        if not len(resolution) == 2:
            raise ValueError(f"Resolution should be a 2d array, but now {len(resolution)} is given.")
        if mount_actor_name is not None:
            mount = [actor for actor in self.scene.get_all_actors() if actor.get_name() == mount_actor_name]
            if len(mount) == 0:
                raise ValueError(f"Camera mount {mount_actor_name} not found in the env.")
            if len(mount) > 1:
                raise ValueError(
                    f"Camera mount {mount_actor_name} name duplicates! To mount an camera on an actor,"
                    f" give the mount a unique name.")
            mount = mount[0]
            cam = self.scene.add_mounted_camera(name, mount, Pose(), width=resolution[0], height=resolution[1],
                                                fovy=fov, fovx=fov, near=0.1, far=10)
        else:
            # Construct camera pose
            look_at_dir = look_at_dir / np.linalg.norm(look_at_dir)
            right_dir = right_dir - np.sum(right_dir * look_at_dir).astype(np.float64) * look_at_dir
            right_dir = right_dir / np.linalg.norm(right_dir)
            up_dir = np.cross(look_at_dir, -right_dir)
            rot_mat_homo = np.stack([look_at_dir, -right_dir, up_dir, position], axis=1)
            pose_mat = np.concatenate([rot_mat_homo, np.array([[0, 0, 0, 1]])])
            pose_cam = sapien.Pose.from_transformation_matrix(pose_mat)
            cam = self.scene.add_camera(name, width=resolution[0], height=resolution[1], fovy=fov, near=0.1, far=10)
            cam.set_local_pose(pose_cam)

        self.cameras.update({name: cam})

    def create_camera_from_pose(self, pose: sapien.Pose, name: str, resolution: Sequence[Union[float, int]],
                                fov: float, use_opencv_trans: bool):
        if not len(resolution) == 2:
            raise ValueError(f"Resolution should be a 2d array, but now {len(resolution)} is given.")
        sapien2opencv = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        sapien2opencv_quat = transforms3d.quaternions.mat2quat(sapien2opencv)
        if use_opencv_trans:
            pose_cam = pose * sapien.Pose(q=sapien2opencv_quat)
        else:
            pose_cam = pose
        cam = self.scene.add_camera(name, width=resolution[0], height=resolution[1], fovy=fov, near=0.1, far=10)
        cam.set_local_pose(pose_cam)
        self.cameras.update({name: cam})

    def setup_camera_from_config(self, config: Dict[str, Dict], use_opencv_trans=True):
        for cam_name, cfg in config.items():
            if cam_name in self.cameras.keys():
                raise ValueError(f"Camera {cam_name} already exists in the environment")
            if "mount_actor_name" in cfg:
                self.create_camera(None, None, None, name=cam_name, resolution=cfg["resolution"],
                                   fov=cfg["fov"], mount_actor_name=cfg["mount_actor_name"])
            else:
                if "position" in cfg:
                    self.create_camera(cfg["position"], cfg["look_at_dir"], cfg["right_dir"], cam_name,
                                       resolution=cfg["resolution"], fov=cfg["fov"])
                elif "pose" in cfg:
                    self.create_camera_from_pose(cfg["pose"], cam_name, resolution=cfg["resolution"], fov=cfg["fov"],
                                                 use_opencv_trans=use_opencv_trans)
                else:
                    raise ValueError(f"Camera {cam_name} has no position or pose.")
