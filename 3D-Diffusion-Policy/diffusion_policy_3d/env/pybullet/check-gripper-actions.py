import os
import pybullet as p
import pybullet_data
import time
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
            )

    def set_gripper(self, gripper_angle):
        """Set gripper angle directly"""
        p.setJointMotorControl2(
            self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=gripper_angle
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


def playback_actions_with_tracking(robot, actions, motion_type="both", sleep_time=0.05):
    """
    Playback actions by maintaining a target state (CORRECT approach)
    
    Key insight: Apply deltas to the INTENDED target state, not actual robot state.
    This prevents drift from accumulating due to tracking error.
    
    Args:
        robot: UR5Robotiq85 robot instance
        actions: (T, 13) array of deltas
        motion_type: "arm", "gripper", or "both"
        sleep_time: time between frames
    """
    print(f"\n{'='*60}")
    print(f"Playing back {len(actions)} action frames")
    print(f"Motion type: {motion_type}")
    print(f"Method: Track target state (prevents drift)")
    print(f"{'='*60}\n")
    
    # Extract joint deltas
    if actions.shape[1] == 13:
        joint_deltas = actions[:, 6:13]
    elif actions.shape[1] == 7:
        joint_deltas = actions
    else:
        raise ValueError(f"Expected 7 or 13 dimensions, got {actions.shape[1]}")
    
    arm_deltas = joint_deltas[:, :6]
    gripper_deltas = joint_deltas[:, 6]
    
    # Initialize target state (what we INTEND the robot to be at)
    initial_joint_pos = robot.get_joint_positions()
    target_arm = initial_joint_pos[:6].copy()
    target_gripper = initial_joint_pos[6]
    
    # Store initial for freezing modes
    initial_arm = target_arm.copy()
    initial_gripper = target_gripper
    
    print(f"Initial target state:")
    print(f"  Arm: {target_arm}")
    print(f"  Gripper: {target_gripper:.6f}\n")
    
    for i in range(len(joint_deltas)):
        # Update target state by applying deltas
        # KEY: Update target, not actual robot position
        if motion_type == "arm":
            target_arm = target_arm + arm_deltas[i]
            robot.set_arm_joints(target_arm)
            robot.set_gripper(initial_gripper)
            
        elif motion_type == "gripper":
            target_gripper = target_gripper + gripper_deltas[i]
            robot.set_arm_joints(initial_arm)
            robot.set_gripper(target_gripper)
            
        else:  # "both"
            target_arm = target_arm + arm_deltas[i]
            target_gripper = target_gripper + gripper_deltas[i]
            robot.set_arm_joints(target_arm)
            robot.set_gripper(target_gripper)
        
        # Step simulation
        for _ in range(5):
            p.stepSimulation()
        
        time.sleep(sleep_time)
        
        # Progress with tracking error monitoring
        if i % 50 == 0 or i == len(joint_deltas) - 1:
            actual_pos = robot.get_joint_positions()
            arm_error = np.linalg.norm(target_arm - actual_pos[:6])
            gripper_error = abs(target_gripper - actual_pos[6])
            
            print(f"Frame {i:4d}/{len(joint_deltas)}: "
                  f"Target_gripper={target_gripper:+.6f}, "
                  f"Actual_gripper={actual_pos[6]:+.6f}, "
                  f"Error={gripper_error:.6f}, "
                  f"Arm_error={arm_error:.6f}")
    
    print(f"\nFinal target state:")
    print(f"  Arm: {target_arm}")
    print(f"  Gripper: {target_gripper:.6f}")


def setup_simulation():
    """Setup PyBullet simulation environment"""
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", [0, 0, 0], useMaximalCoordinates=True)
    p.loadURDF("table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))


def main():
    print("=" * 60)
    print("Robot Action Playback - Fixed Version")
    print("Solution 2: Track Target State (No Drift)")
    print("=" * 60)
    
    actions_file = "pred-act-formatted.txt"
    
    if not os.path.exists(actions_file):
        print(f"Error: File not found: {actions_file}")
        return
    
    print("\nSelect motion type:")
    print("1. Arm only motion")
    print("2. Gripper only motion")
    print("3. Arm + Gripper motion")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    motion_map = {"1": "arm", "2": "gripper", "3": "both"}
    motion_type = motion_map.get(choice, "both")
    
    # Load actions
    print(f"\nLoading actions from: {actions_file}")
    try:
        actions = load_actions(actions_file)
        print(f"Loaded {len(actions)} action frames")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Setup simulation
    setup_simulation()
    
    # Load robot
    robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
    robot.load()
    
    # Move to rest pose
    rest_pose = [0, -1.57, 1.57, -1.5, -1.57, 0.0]
    for i, joint_id in enumerate(robot.arm_controllable_joints):
        p.setJointMotorControl2(
            robot.id, joint_id, p.POSITION_CONTROL, rest_pose[i]
        )
    p.setJointMotorControl2(
        robot.id, 
        robot.mimic_parent_id,
        p.POSITION_CONTROL,
        targetPosition=0.0,
        force=200
    )
    
    # Stabilize
    print("\nStabilizing robot at rest pose...")
    for _ in range(1000):
        p.stepSimulation()
    
    print("Starting playback...\n")
    
    # Playback with proper tracking
    playback_actions_with_tracking(robot, actions, motion_type=motion_type)
    
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