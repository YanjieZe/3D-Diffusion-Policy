import os
import pybullet as p
import pybullet_data
import math
import time
import random
from collections import namedtuple
import numpy as np


class UR5Robotiq85:
    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]
        self.gripper_range = [0, 0.085]
        self.gripper_angle_range = [0.0, 0.8]
        self.max_velocity = 3

    def load(self):
        self.id = p.loadURDF(
            "./urdf/ur5_robotiq_85.urdf",
            self.base_pos,
            self.base_ori,
            useFixedBase=True,
        )
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()

    def __parse_joint_info__(self):
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

    def __setup_mimic_joints__(self):
        mimic_parent_name = "finger_joint"
        mimic_children_names = {
            "right_outer_knuckle_joint": 1,
            "left_inner_knuckle_joint": 1,
            "right_inner_knuckle_joint": 1,
            "left_inner_finger_joint": -1,
            "right_inner_finger_joint": -1,
        }
        self.mimic_parent_id = [
            joint.id for joint in self.joints if joint.name == mimic_parent_name
        ][0]
        self.mimic_child_multiplier = {
            joint.id: mimic_children_names[joint.name]
            for joint in self.joints
            if joint.name in mimic_children_names
        }
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
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

    def get_joint_positions(self):
        """Get current joint positions (6 arm + 1 gripper)"""
        joint_positions = []
        for joint_id in self.arm_controllable_joints:
            joint_state = p.getJointState(self.id, joint_id)
            joint_positions.append(joint_state[0])
        
        gripper_state = p.getJointState(self.id, self.mimic_parent_id)
        joint_positions.append(gripper_state[0])
        
        return np.array(joint_positions)

    def set_arm_joints(self, joint_positions):
        """Set arm joint positions directly"""
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(
                self.id,
                joint_id,
                p.POSITION_CONTROL,
                joint_positions[i],
                maxVelocity=self.max_velocity,
                force=500,
            )

    def set_gripper(self, gripper_angle):
        """Set gripper angle directly"""
        gripper_angle = np.clip(gripper_angle, self.gripper_angle_range[0], self.gripper_angle_range[1])
        p.setJointMotorControl2(
            self.id, 
            self.mimic_parent_id, 
            p.POSITION_CONTROL, 
            targetPosition=gripper_angle,
            force=200,
        )

    def set_joint_positions(self, joint_positions):
        """Set all joint positions (6 arm + 1 gripper)"""
        self.set_arm_joints(joint_positions[:6])
        if len(joint_positions) > 6:
            self.set_gripper(joint_positions[6])


def load_actions(file_path):
    """Load action deltas from text file"""
    actions = np.loadtxt(file_path)
    return actions


def playback_actions(robot, actions, motion_type="both", sleep_time=0.05):
    """
    Playback recorded actions (deltas) from data collection script
    
    Args:
        robot: UR5Robotiq85 robot instance
        actions: numpy array of action deltas, shape (T, 13)
                 Format: [eef_pos(3), eef_orn(3), arm_joint_deltas(6), gripper_delta(1)]
        motion_type: "arm", "gripper", or "both"
        sleep_time: time between frames for visualization
    """
    print(f"\n{'='*60}")
    print(f"Playing back {len(actions)} action frames")
    print(f"Motion type: {motion_type}")
    print(f"Action shape: {actions.shape}")
    print(f"{'='*60}\n")
    
    # Extract joint deltas from 13D actions
    # Format: [eef_pos(3), eef_orn(3), arm_joints(6), gripper(1)]
    # We want indices 6:13 (arm_joints + gripper)
    if actions.shape[1] == 13:
        joint_deltas = actions[:, 6:13]  # Extract arm_joints(6) + gripper(1)
        print("Extracted joint deltas from 13D actions (indices 6:13)")
    elif actions.shape[1] == 7:
        joint_deltas = actions
        print("Using 7D actions directly")
    else:
        raise ValueError(f"Expected actions with 7 or 13 dimensions, got {actions.shape[1]}")
    
    # Split into arm and gripper
    arm_deltas = joint_deltas[:, :6]
    gripper_deltas = joint_deltas[:, 6]
    
    print(f"Arm deltas range: [{arm_deltas.min():.4f}, {arm_deltas.max():.4f}]")
    print(f"Gripper deltas range: [{gripper_deltas.min():.6f}, {gripper_deltas.max():.6f}]")
    
    # Get initial positions
    current_joint_pos = robot.get_joint_positions()
    initial_arm_pos = current_joint_pos[:6].copy()
    initial_gripper_pos = current_joint_pos[6]
    
    print(f"\nStarting from rest pose:")
    print(f"  Arm joints: {initial_arm_pos}")
    print(f"  Gripper: {initial_gripper_pos:.4f}")
    print(f"\nStarting playback...\n")
    
    # Apply actions frame by frame
    for i in range(len(joint_deltas)):
        # Get current state
        current_joint_pos = robot.get_joint_positions()
        
        if motion_type == "arm":
            # Move arm, freeze gripper at initial position
            new_arm_pos = current_joint_pos[:6] + arm_deltas[i]
            new_gripper_pos = initial_gripper_pos
            
        elif motion_type == "gripper":
            # Freeze arm at initial position, move gripper
            new_arm_pos = initial_arm_pos
            new_gripper_pos = current_joint_pos[6] + gripper_deltas[i]
            
        else:  # "both"
            # Move both arm and gripper
            new_arm_pos = current_joint_pos[:6] + arm_deltas[i]
            new_gripper_pos = current_joint_pos[6] + gripper_deltas[i]
        
        # Combine and apply
        new_joint_pos = np.concatenate([new_arm_pos, [new_gripper_pos]])
        robot.set_joint_positions(new_joint_pos)
        
        # Step simulation for smooth motion
        for _ in range(5):
            p.stepSimulation()
        
        # Slow down for visualization
        time.sleep(sleep_time)
        
        # Progress update every 50 frames
        if i % 50 == 0 or i == len(joint_deltas) - 1:
            actual_pos = robot.get_joint_positions()
            print(f"Frame {i:4d}/{len(joint_deltas)}: "
                  f"Gripper={actual_pos[6]:.4f}, "
                  f"Gripper_delta={gripper_deltas[i]:+.6f}")


def setup_simulation():
    """Setup PyBullet simulation environment"""
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", [0, 0, 0], useMaximalCoordinates=True)
    p.loadURDF("table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))
    
    # Load tray (optional, for visualization)
    tray_pos = [0.5, 0.9, 0.6]
    p.loadURDF("tray/tray.urdf", tray_pos, p.getQuaternionFromEuler([0, 0, 0]))


def main():
    print("=" * 60)
    print("Robot Action Playback - Data Collection Format")
    print("=" * 60)
    
    # Get actions file path
    default_path = "iter_0000.txt"
    actions_file = default_path
    
    # Check if file exists
    if not os.path.exists(actions_file):
        print(f"Error: File not found: {actions_file}")
        return
    
    # Select motion type
    print("\nSelect motion type:")
    print("1. Arm only motion (freeze gripper)")
    print("2. Gripper only motion (freeze arm)")
    print("3. Arm + Gripper motion (full trajectory)")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    motion_map = {
        "1": "arm",
        "2": "gripper",
        "3": "both"
    }
    
    motion_type = motion_map.get(choice, "both")
    
    # Load actions
    print(f"\nLoading actions from: {actions_file}")
    try:
        actions = load_actions(actions_file)
        print(f"Loaded {len(actions)} action frames with shape {actions.shape}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Setup simulation
    print("\nSetting up simulation...")
    setup_simulation()
    
    # Load robot
    robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
    robot.load()
    
    # Set to rest pose (matching data collection initial state)
    # Rest pose: arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0], gripper = 0.0
    initial_joint_positions = np.array([0, -1.57, 1.57, -1.5, -1.57, 0.0, 0.0])
    robot.set_joint_positions(initial_joint_positions)
    
    # Wait for simulation to stabilize
    for _ in range(100):
        p.stepSimulation()
    
    # Verify initial state
    actual_initial = robot.get_joint_positions()
    print(f"\nRobot initialized to rest pose:")
    print(f"  Target: {initial_joint_positions}")
    print(f"  Actual: {actual_initial}")
    
    # Playback actions
    playback_actions(robot, actions, motion_type=motion_type, sleep_time=0.05)
    
    print(f"\n{'='*60}")
    print("Playback complete!")
    print("Press Ctrl+C to exit")
    print(f"{'='*60}\n")
    
    # Keep simulation running
    try:
        while True:
            p.stepSimulation()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nExiting...")
        p.disconnect()


if __name__ == "__main__":
    main()