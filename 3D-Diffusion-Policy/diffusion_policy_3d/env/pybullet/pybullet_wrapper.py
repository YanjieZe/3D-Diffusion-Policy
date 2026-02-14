import gym
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import pybullet as p
import pybullet_data
import cv2
from termcolor import cprint
from gym import spaces
import random
import time


def compute_workspace_bounds(pc_xyz, n_std=30):
    """
    Compute workspace bounds using mean ± n_std * std
    Args:
        pc_xyz: (N, 3) numpy array of point cloud coordinates
        n_std: number of standard deviations for bounds
    Returns:
        WORK_SPACE: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    """
    if pc_xyz.shape[0] == 0:
        return None

    mean = pc_xyz.mean(axis=0)
    std = pc_xyz.std(axis=0)

    WORK_SPACE = [
        [mean[0] - n_std * std[0], mean[0] + n_std * std[0]],  # X
        [mean[1] - n_std * std[1], mean[1] + n_std * std[1]],  # Y
        [mean[2] - n_std * std[2], mean[2] + n_std * std[2]],  # Z
    ]

    return WORK_SPACE


def crop_workspace(pc_xyz, workspace_bounds):
    """
    Crop point cloud to workspace bounds
    Args:
        pc_xyz: (N, 3) numpy array
        workspace_bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    Returns:
        mask: boolean array of valid points
    """
    if workspace_bounds is None:
        return np.ones(pc_xyz.shape[0], dtype=bool)

    mask = (
        (pc_xyz[:, 0] >= workspace_bounds[0][0])
        & (pc_xyz[:, 0] <= workspace_bounds[0][1])
        & (pc_xyz[:, 1] >= workspace_bounds[1][0])
        & (pc_xyz[:, 1] <= workspace_bounds[1][1])
        & (pc_xyz[:, 2] >= workspace_bounds[2][0])
        & (pc_xyz[:, 2] <= workspace_bounds[2][1])
    )

    return mask


def downsample_with_fps(points: np.ndarray, num_points: int = 6000):
    """Fast point cloud sampling using torch3d"""
    if points.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    points = torch.from_numpy(points).unsqueeze(0).cuda()
    num_points_tensor = torch.tensor([num_points]).cuda()

    # remember to only use coord to sample
    _, sampled_indices = torch3d_ops.sample_farthest_points(
        points=points[..., :3], K=num_points_tensor
    )

    points = points.squeeze(0).cpu().numpy()
    points = points[sampled_indices.squeeze(0).cpu().numpy()]

    return points


def depth_to_point_cloud(
    depth_buffer,
    view_matrix,
    proj_matrix,
    width=224,
    height=224,
    fov=60,
    near=0.01,
    far=3.0,
    max_depth=2.5,
    exclude_mask=None,
):
    """
    Convert a PyBullet depth buffer to a 3D point cloud in world coordinates,
    optionally filtering points beyond max_depth.

    Args:
        depth_buffer: depth buffer from PyBullet (H x W)
        view_matrix: PyBullet camera view matrix (16-element list)
        proj_matrix: PyBullet projection matrix (16-element list)
        width: image width
        height: image height
        fov: field of view (degrees)
        near: near clipping plane
        far: far clipping plane
        max_depth: maximum distance to keep points (in meters)
    """
    # Convert depth buffer to linear depth
    depth_img = far * near / (far - (far - near) * depth_buffer)

    # Camera intrinsics
    fx = fy = width / (2 * np.tan(np.radians(fov) / 2))
    cx, cy = width / 2, height / 2

    # Pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # 3D points in camera frame
    z = depth_img
    x = -(u - cx) * z / fx
    y = -(v - cy) * z / fy  # negative to flip y-axis

    points_camera = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Filter points beyond max_depth
    valid_mask = points_camera[:, 2] < max_depth

    if exclude_mask is not None:  # to exclude table and plane
        valid_mask = valid_mask & (~exclude_mask)

    points_camera = points_camera[valid_mask]
    return points_camera


class UR5Robotiq85:
    """UR5 Robot with Robotiq 85 Gripper"""

    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]
        self.gripper_range = [0, 0.085]
        self.max_velocity = 3

        # CRITICAL: Gripper normalization limit (same as data collection)
        self.GRIPPER_NORM_LIMIT = 0.2
        self.current_gripper_velocity = 0.0  # Track current velocity command


    def load(self):
        self.id = p.loadURDF(
            "/home/cross-emb/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env/pybullet/urdf/ur5_robotiq_85.urdf",
            self.base_pos,
            self.base_ori,
            useFixedBase=True,
        )
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()

    def __parse_joint_info__(self):
        from collections import namedtuple

        jointInfo = namedtuple(
            "jointInfo",
            [
                "id",
                "name",
                "type",
                "lowerLimit",
                "upperLimit",
                "maxForce",
                "maxVelocity",
                "controllable",
            ],
        )
        self.joints = []
        self.controllable_joints = []

        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED

            if controllable:
                self.controllable_joints.append(jointID)

            self.joints.append(
                jointInfo(
                    jointID,
                    jointName,
                    jointType,
                    jointLowerLimit,
                    jointUpperLimit,
                    jointMaxForce,
                    jointMaxVelocity,
                    controllable,
                )
            )

        self.arm_controllable_joints = self.controllable_joints[: self.arm_num_dofs]
        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][
            : self.arm_num_dofs
        ]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][
            : self.arm_num_dofs
        ]
        self.arm_joint_ranges = [
            ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)
        ]

    def __setup_mimic_joints__(self):
        """Setup mimic joints for Robotiq gripper with STRONG constraints"""
        mimic_parent_name = "finger_joint"
        mimic_children_names = {
            "right_outer_knuckle_joint": 1,
            "left_inner_knuckle_joint": 1,
            "right_inner_knuckle_joint": 1,
            "left_inner_finger_joint": -1,
            "right_inner_finger_joint": -1,
        }
        
        # Find parent joint ID
        self.mimic_parent_id = [
            joint.id for joint in self.joints if joint.name == mimic_parent_name
        ][0]
        
        # Store child joint info
        self.mimic_child_multiplier = {
            joint.id: mimic_children_names[joint.name]
            for joint in self.joints
            if joint.name in mimic_children_names
        }
        
        # Create constraints with STRONG force (same as data collection)
        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(
                self.id,
                self.mimic_parent_id,
                self.id,
                joint_id,
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            # CRITICAL: High maxForce and erp=1 for stiff constraints
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

            
    def get_robot_state(self, include_gripper=True):
        """
        Get complete robot state
        
        Args:
            include_gripper: If True, return [eef_pos(3), eef_orn_euler(3), joint_angles(6), gripper_normalized(1)]
                           If False, return [eef_pos(3), eef_orn_euler(3), joint_angles(6)]
        
        Returns: 
            state array of size 13 (with gripper) or 12 (without gripper)
        """
        eef_state = p.getLinkState(self.id, self.eef_id)
        eef_pos = np.array(eef_state[0])
        eef_orn_quat = np.array(eef_state[1])
        eef_orn_euler = np.array(p.getEulerFromQuaternion(eef_orn_quat))

        joint_states = []
        for joint_id in self.arm_controllable_joints:
            joint_state = p.getJointState(self.id, joint_id)
            joint_states.append(joint_state[0])

        if include_gripper:
            # Get raw gripper angle and normalize
            raw_angle = p.getJointState(self.id, self.mimic_parent_id)[0]
            
            # Noise suppression
            if abs(raw_angle) < 1e-3:
                raw_angle = 0.0
            
            # Normalize to [0, 1] using GRIPPER_NORM_LIMIT
            gripper_normalized = np.clip(raw_angle / self.GRIPPER_NORM_LIMIT, 0.0, 1.0)
            state = np.concatenate([eef_pos, eef_orn_euler, joint_states, [gripper_normalized]])
        else:
            state = np.concatenate([eef_pos, eef_orn_euler, joint_states])

        return state
    

    def get_joint_positions(self, include_gripper=True):
        """
        Get current joint positions
        
        Args:
            include_gripper: If True, return [arm_joints(6), gripper_normalized(1)]
                           If False, return [arm_joints(6)]
        
        Returns:
            joint positions array
        """
        joint_positions = []
        for joint_id in self.arm_controllable_joints:
            joint_state = p.getJointState(self.id, joint_id)
            joint_positions.append(joint_state[0])

        if include_gripper:
            # Get normalized gripper value
            raw_angle = p.getJointState(self.id, self.mimic_parent_id)[0]
            if abs(raw_angle) < 1e-3:
                raw_angle = 0.0
            gripper_normalized = np.clip(raw_angle / self.GRIPPER_NORM_LIMIT, 0.0, 1.0)
            joint_positions.append(gripper_normalized)

        return np.array(joint_positions)


    def get_joint_limits(self, include_gripper=True):
        """
        Get joint limits
        
        Args:
            include_gripper: If True, include gripper limits
        
        Returns:
            limits_lower, limits_upper
        """
        limits_lower = self.arm_lower_limits
        limits_upper = self.arm_upper_limits
        
        if include_gripper:
            limits_lower = limits_lower + [self.gripper_range[0]]
            limits_upper = limits_upper + [self.gripper_range[1]]
            
        return np.array(limits_lower), np.array(limits_upper)

    def set_arm_joints(self, joint_positions):
        """Set arm joint positions directly"""
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(
                self.id,
                joint_id,
                p.POSITION_CONTROL,
                joint_positions[i],
                maxVelocity=self.max_velocity,
            )

    def move_gripper(self, velocity_command):
        """
        Velocity-based gripper control (same as data collection)
        
        Args:
            velocity_command: Discrete control input
                -1.0: Open gripper (executed as -0.25 for safety)
                 0.0: Hold current state
                 1.0: Close gripper
        """
        # Store the velocity command
        self.current_gripper_velocity = velocity_command
        
        # Map discrete command to actual velocity
        if velocity_command == -1.0:
            actual_velocity = -0.25 * 20.0  # Open slowly
        elif velocity_command == 0.0:
            actual_velocity = 0.0  # Hold
        elif velocity_command == 1.0:
            actual_velocity = 1.0 * 20.0  # Close
        else:
            # Continuous value - scale it
            cprint("Warning: Using continuous gripper velocity command.", "red")
            actual_velocity = velocity_command * 20.0
        
        p.setJointMotorControl2(
            self.id, 
            self.mimic_parent_id, 
            p.VELOCITY_CONTROL,
            targetVelocity=actual_velocity,
            force=100  # High force for reliable movement
        )

            
    def set_joint_positions(self, joint_positions, include_gripper=True):
        """
        Set all joint positions
        
        Args:
            joint_positions: If include_gripper=True: [arm_joints(6), gripper_velocity_cmd(1)]
                           If include_gripper=False: [arm_joints(6)]
            include_gripper: Whether gripper command is included
        """
        # Set arm joints
        self.set_arm_joints(joint_positions[:6])
        
        # Set gripper using velocity command (if included)
        if include_gripper and len(joint_positions) > 6:
            gripper_cmd = joint_positions[6]
            self.move_gripper(gripper_cmd)


    def get_eef_position(self):
        """Get end-effector 3D position"""
        eef_state = p.getLinkState(self.id, self.eef_id)
        return np.array(eef_state[0])


class UR5PickPlaceEnv(gym.Env):
    """
    PyBullet UR5 Pick and Place Environment
    
    Supports both 6D (arm only) and 7D (arm + gripper) action spaces
    """

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(
        self,
        use_gui=False,
        num_points=6000,
        image_size=224,
        use_workspace_crop=True,
        workspace_std=30.0,
        action_dim=7,
        capture_table=False,
    ):
        self.use_gui = use_gui
        self.num_points = num_points
        self.image_size = image_size
        self.max_steps = 350
        self.current_step = 0

        self.use_workspace_crop = use_workspace_crop
        self.workspace_std = workspace_std
        self.workspace_bounds = None

        self.action_dim = action_dim
        self.capture_table = capture_table
        
        # NEW: Determine if gripper is included
        self.include_gripper = (action_dim == 7 or action_dim == 13)
        
        cprint(f"Environment initialized with action_dim={action_dim}", "cyan")
        cprint(f"Include gripper: {self.include_gripper}", "cyan")

        # Connect to PyBullet
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.table_id = p.loadURDF(
            "table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0])
        )

        self.tray_pos = [0.5, 0.9, 0.6]
        self.tray_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.tray_id = p.loadURDF("tray/tray.urdf", self.tray_pos, self.tray_orn)

        # Load robot
        self.robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
        self.robot.load()

        # Camera parameters
        self.tp_cam_eye = [1.1, -0.6, 1.3]
        self.tp_cam_target = [0.5, 0.3, 0.7]
        self.tp_cam_up = [0, 0, 1]

        self.cube_id = None
        self.cube_start_pos = None

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(self.action_dim,), dtype=np.float32
        )

        self.observation_space = spaces.Dict(
            {
                "point_cloud": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_points, 3),
                    dtype=np.float32,
                ),
                "agent_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.action_dim,), dtype=np.float32
                ),
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, self.image_size, self.image_size),
                    dtype=np.uint8,
                ),
            }
        )

        self.is_success_flag = False


    def reset(self, cube_start_pos=None, cube_start_orn=None):
        """
        Reset the environment

        Args:
            cube_start_pos: Optional 3D position [x, y, z] for cube. If None, random position is used.
            cube_start_orn: Optional quaternion [x, y, z, w] for cube orientation. If None, [0, 0, 0, 1] is used.
        """
        cprint(f"IN ENV RESET parameters {cube_start_pos} and {cube_start_orn}", "cyan")

        self.current_step = 0
        self.is_success_flag = False
        self.workspace_bounds = None

        # Remove old cube if exists
        if self.cube_id is not None:
            p.removeBody(self.cube_id)

        # Move to rest pose
        rest_pose = [0, -1.57, 1.57, -1.5, -1.57, 0.0]
        for i, joint_id in enumerate(self.robot.arm_controllable_joints):
            p.setJointMotorControl2(
                self.robot.id, joint_id, p.POSITION_CONTROL, rest_pose[i]
            )

        if self.include_gripper:
            p.setJointMotorControl2(
                self.robot.id, 
                self.robot.mimic_parent_id, 
                p.POSITION_CONTROL, 
                targetPosition=0.000,
                force=1500,
                maxVelocity=1.5
            )
    
        for _ in range(5000):
            p.stepSimulation()
        
        # Stabilize
        print("\nStabilizing robot at rest pose...")
        for _ in range(1000):
            p.stepSimulation()

        if self.include_gripper:
            actual_angle = p.getJointState(self.robot.id, self.robot.mimic_parent_id)[0]
            print("Reset complete. Gripper position after reset: {:.4f}\n".format(actual_angle))

        # Use provided cube position or generate random one
        if cube_start_pos is not None:
            self.cube_start_pos = cube_start_pos
        else:
            cprint("Randomizing cube start position.", "red")
            self.cube_start_pos = [
                random.uniform(0.3, 0.7),
                random.uniform(-0.1, 0.1),
                0.65,
            ]

        # Use provided orientation or default to identity quaternion
        if cube_start_orn is not None:
            if len(cube_start_orn) == 4:
                cube_orn_quat = cube_start_orn
            else:
                cube_orn_quat = p.getQuaternionFromEuler(cube_start_orn)
        else:
            cprint("Using default cube(random) orientation.", "red")
            cube_orn_quat = p.getQuaternionFromEuler([0, 0, 0])

        self.cube_id = p.loadURDF("cube_small.urdf", self.cube_start_pos, cube_orn_quat)

        # Random cube color
        color = [random.random(), random.random(), random.random(), 1.0]
        p.changeVisualShape(self.cube_id, -1, rgbaColor=color)

        # Step simulation
        for _ in range(50):
            p.stepSimulation()

        obs = self._get_obs()
        return obs

    def step(self, action):
        """
        Execute action and return observation.
        
        Args:
            action: Action array. Can be:
                - 6D: [arm_deltas(6)] - arm only, no gripper
                - 7D: [arm_deltas(6), gripper_velocity_cmd(1)]
                - 13D: [eef_delta(6), arm_deltas(6), gripper_velocity_cmd(1)]
        """
        self.current_step += 1

        # Handle different action dimensions
        if len(action) == 13:
            joint_deltas = action[6:13]
            arm_deltas = joint_deltas[:6]
            if self.include_gripper:
                gripper_velocity_cmd = joint_deltas[6]
        elif len(action) == 7:
            arm_deltas = action[:6]
            if self.include_gripper:
                gripper_velocity_cmd = action[6]
        elif len(action) == 6:
            # 6D action - arm only, no gripper
            arm_deltas = action[:6]
            gripper_velocity_cmd = None
        else:
            raise ValueError(f"Action must be 6D, 7D, or 13D, got {len(action)}D")

        # Discretize gripper command if included
        if self.include_gripper and gripper_velocity_cmd is not None:
            if gripper_velocity_cmd < -0.33:
                gripper_cmd_discrete = -1.0
            elif gripper_velocity_cmd > 0.33:
                gripper_cmd_discrete = 1.0
            else:
                gripper_cmd_discrete = 0.0
        else:
            gripper_cmd_discrete = None

        # Get current state
        current_joint_pos = self.robot.get_joint_positions(include_gripper=self.include_gripper)

        # Update arm joints
        target_arm = current_joint_pos[:6] + arm_deltas
        self.robot.set_arm_joints(target_arm)

        # Update gripper if included
        if self.include_gripper and gripper_cmd_discrete is not None:
            self.robot.move_gripper(gripper_cmd_discrete)

        for _ in range(1000):
            p.stepSimulation()

        time.sleep(0.55)

        obs = self._get_obs()

        # Check success: cube in tray
        reward = 0.0
        done = False
        if self.cube_id is not None:
            cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
            if (
                abs(cube_pos[0] - self.tray_pos[0]) < 0.2
                and abs(cube_pos[1] - self.tray_pos[1]) < 0.2
                and abs(cube_pos[2] - self.tray_pos[2]) < 0.1
            ):
                reward = 1.0
                self.is_success_flag = True
                done = True

        # Episode timeout
        if self.current_step >= self.max_steps:
            done = True

        info = {"is_success": self.is_success_flag}
        return obs, reward, done, info

    def _get_obs(self):
        """Get current observation"""
        robot_state = self.robot.get_robot_state(include_gripper=self.include_gripper)

        # Extract appropriate agent_pos based on action_dim
        if self.action_dim == 6:
            # 6D: [eef_pos(3), eef_orn_euler(3), joint_angles(6)] -> use last 6 (joint angles)
            agent_pos = robot_state[6:12]
        elif self.action_dim == 7:
            # 7D: [joint_angles(6), gripper(1)]
            agent_pos = robot_state[6:13]
        elif self.action_dim == 13:
            # 13D: full state
            agent_pos = robot_state
        else:
            raise ValueError(
                f"Unsupported action_dim: {self.action_dim}. Must be 6, 7, or 13."
            )

        # Exclude table from point cloud
        exclude_ids = []
        if self.table_id is not None and not self.capture_table:
            exclude_ids.append(self.table_id)

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.tp_cam_eye,
            cameraTargetPosition=self.tp_cam_target,
            cameraUpVector=self.tp_cam_up,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.01, farVal=3.0
        )

        # Capture with segmentation
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            self.image_size,
            self.image_size,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        )

        rgb_img = np.array(rgb_img)[:, :, :3]
        depth_buffer = np.array(depth_img)
        seg_img = np.array(seg_img)

        exclude_mask_tp = np.zeros_like(seg_img, dtype=bool)
        for obj_id in exclude_ids:
            exclude_mask_tp |= seg_img == obj_id

        exclude_mask_flat_tp = exclude_mask_tp.flatten()

        point_cloud = depth_to_point_cloud(
            depth_buffer,
            view_matrix,
            proj_matrix,
            self.image_size,
            self.image_size,
            exclude_mask=exclude_mask_flat_tp,
        )

        # Workspace cropping
        if self.use_workspace_crop:
            if self.workspace_bounds is None:
                self.workspace_bounds = compute_workspace_bounds(
                    point_cloud, n_std=self.workspace_std
                )
                if self.workspace_bounds is not None:
                    print(
                        f"Computed workspace bounds (mean ± {self.workspace_std}*std):"
                    )
                    print(
                        f"  X: [{self.workspace_bounds[0][0]:.3f}, {self.workspace_bounds[0][1]:.3f}]"
                    )
                    print(
                        f"  Y: [{self.workspace_bounds[1][0]:.3f}, {self.workspace_bounds[1][1]:.3f}]"
                    )
                    print(
                        f"  Z: [{self.workspace_bounds[2][0]:.3f}, {self.workspace_bounds[2][1]:.3f}]"
                    )

            if self.workspace_bounds is not None:
                mask = crop_workspace(point_cloud, self.workspace_bounds)
                point_cloud = point_cloud[mask]
                if self.current_step == 0:
                    print(f"  After workspace crop: {point_cloud.shape[0]} points")

        # Downsample or pad to target number of points
        if point_cloud.shape[0] == 0:
            point_cloud = np.zeros((self.num_points, 3), dtype=np.float32)
        elif point_cloud.shape[0] > self.num_points:
            point_cloud = downsample_with_fps(point_cloud, self.num_points)
        elif point_cloud.shape[0] < self.num_points:
            padding = np.zeros((self.num_points - point_cloud.shape[0], 3))
            point_cloud = np.vstack([point_cloud, padding])

        rgb_img = rgb_img.transpose(2, 0, 1)

        obs_dict = {
            "point_cloud": point_cloud.astype(np.float32),
            "agent_pos": agent_pos.astype(np.float32),
            "image": rgb_img.astype(np.uint8),
        }

        return obs_dict

    def render(self, mode="rgb_array"):
        """Render the environment"""
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.tp_cam_eye,
            cameraTargetPosition=self.tp_cam_target,
            cameraUpVector=self.tp_cam_up,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.01, farVal=3.0
        )

        width, height, rgb_img, _, _ = p.getCameraImage(
            self.image_size,
            self.image_size,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
        )

        rgb_img = np.array(rgb_img)[:, :, :3]
        return rgb_img.astype(np.uint8)

    def close(self):
        """Close the environment"""
        p.disconnect(self.physics_client)

    def is_success(self):
        """Check if task was successful"""
        return self.is_success_flag

    def seed(self, seed=None):
        """Set random seed"""
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)