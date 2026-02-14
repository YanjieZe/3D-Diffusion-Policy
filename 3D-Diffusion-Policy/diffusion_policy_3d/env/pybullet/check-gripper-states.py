import numpy as np
import pybullet as p
import pybullet_data
from collections import namedtuple
import time


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


def load_robot_states(file_path):
    """Load robot states from text file"""
    # Each line has 13 values: [eef_pos(3), eef_orn(3), arm_joints(6), gripper(1)]
    states = np.loadtxt(file_path)
    return states


def playback_motion(robot, states, motion_type="both", sleep_time=0.05):
    """
    Playback recorded motion
    """
    print(f"\nPlaying back {len(states)} frames with motion_type='{motion_type}'")

    arm_joints = states[:, 6:12]
    gripper_angles = states[:, 12]

    initial_arm_joints = arm_joints[0]
    initial_gripper = gripper_angles[0]

    for i in range(len(states)):
        if motion_type == "arm":
            robot.set_arm_joints(arm_joints[i])
            robot.set_gripper(initial_gripper)

        elif motion_type == "gripper":
            robot.set_arm_joints(initial_arm_joints)
            robot.set_gripper(gripper_angles[i])

        else:  # both
            robot.set_arm_joints(arm_joints[i])
            robot.set_gripper(gripper_angles[i])

        # Physics steps for smoothness
        for _ in range(5):
            p.stepSimulation()

        # 🔹 Slow down visualization
        time.sleep(sleep_time)

        if i % 50 == 0:
            print(f"Frame {i}/{len(states)}")

def setup_simulation():
    """Setup PyBullet simulation environment"""
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", [0, 0, 0], useMaximalCoordinates=True)
    p.loadURDF("table/table.urdf", [0.5, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))


def main():
    # User input
    print("=" * 60)
    print("Robot Motion Playback")
    print("=" * 60)
    
    # File path
    file_path =  "/home/aniruth/Desktop/RRC/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/env/pybullet/states.txt"
    
    # Motion type selection
    print("\nSelect motion type:")
    print("1. Arm only motion")
    print("2. Gripper only motion")
    print("3. Arm + Gripper motion")
    
    choice = input("Enter choice (1/2/3): ")
    
    motion_map = {
        "1": "arm",
        "2": "gripper",
        "3": "both"
    }
    
    motion_type = motion_map.get(choice, "both")
    
    print(f"\nLoading states from: {file_path}")
    print(f"Motion type: {motion_type}")
    
    # Load states
    try:
        states = load_robot_states(file_path)
        print(f"Loaded {len(states)} states")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Setup simulation
    setup_simulation()
    
    # Load robot
    robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
    robot.load()


    target_joint_positions = [0, -1.57, 1.57, -1.5, -1.57, 0.0]
    for i, joint_id in enumerate(robot.arm_controllable_joints):
        p.setJointMotorControl2(
            robot.id, joint_id, p.POSITION_CONTROL, target_joint_positions[i]
        )
    p.setJointMotorControl2(
        robot.id, 
        robot.mimic_parent_id,
        p.POSITION_CONTROL,
        targetPosition=0.0000,
        force=200
    )
    
        
    # Step simulation to stabilize
    for _ in range(5000):
        p.stepSimulation()


    # Playback motion
    playback_motion(robot, states, motion_type=motion_type)
    
    print("\nPlayback complete! Press Ctrl+C to exit.")
    
    # Keep simulation running
    try:
        while True:
            p.stepSimulation()
    except KeyboardInterrupt:
        print("\nExiting...")
        p.disconnect()


if __name__ == "__main__":
    main()