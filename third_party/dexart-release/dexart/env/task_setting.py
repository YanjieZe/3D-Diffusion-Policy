import sapien.core as sapien
import transforms3d
import numpy as np

ROBOT_TABLE_MARGIN_X = 0.06
ROBOT_TABLE_MARGIN_Y = 0.04

BOUND_CONFIG = {
    "faucet": [0.1, 2.0, -1.0, 1, -0.1352233 + 0.14, 0.4],
    "bucket": [0.1, 2.0, -2.0, 2.0, -0.29, 0.4],
    "laptop": [0.1, 1.0, -1.0, 2, -0.1352233 + 0.14, 0.6],
    "toilet": [0.1, 2.0, -2.0, 2, -0.3, 0.8],
}

ROBUSTNESS_INIT_CAMERA_CONFIG = {
    'laptop': {'r': 1, 'phi': np.pi / 2, 'theta': np.pi / 2,
               'center': np.array([0, 0, 0.5])},
}

TRAIN_CONFIG = {
    "faucet": {
        'seen': [148, 693, 822, 857, 991, 1011, 1053, 1288, 1343, 1370, 1466],
        'unseen': [1556, 1633, 1646, 1667, 1741, 1832, 1925]
    },
    "faucet_half": {
        'seen': [148, 693, 822, 857, 991],
        'unseen': [1556, 1633, 1646, 1667, 1741, 1832, 1925]
    },
    "bucket": {
        'seen': [100431, 100435, 100438, 100439, 100441, 100444, 100446, 100448, 100454, 100461, 100462],
        'unseen': [100468, 100470, 100473, 100482, 100484, 100486, 102352, 102358]
    },
    "bucket_half": {
        'seen': [100431, 100435, 100438, 100439, 100441],
        'unseen': [100468, 100470, 100473, 100482, 100484, 100486, 102352, 102358]
    },
    "laptop": {
        'seen': [11395, 11405, 11406, 11477, 11581, 11586, 9996, 10090, 10098, 10101, 10125],
        'unseen': [9748, 9912, 9918, 9960, 9968, 9992],
    },
    "laptop_half": {
        'seen': [11395, 11405, 11406, 11477, 11581],
        'unseen': [9748, 9912, 9918, 9960, 9968, 9992],
    },
    "toilet": {
        'seen': [102677, 102687, 102689, 102692, 102697, 102699, 102701, 102703, 102707, 102708, 103234, 102663, 102666,
                 102667, 102669, 102670, 102675],
        'unseen': [101320, 102621, 102622, 102630, 102634, 102645, 102648, 102651, 102652, 102654, 102658],
    },
    "toilet_half": {
        'seen': [102677, 102687, 102689, 102692, 102697, 102699, 102701, 102703],
        'unseen': [101320, 102621, 102622, 102630, 102634, 102645, 102648, 102651, 102652, 102654, 102658],
    },
}

TASK_CONFIG = {
    "faucet": [148, 693, 822, 857, 991, 1011, 1053, 1288, 1343, 1370, 1466, 1556, 1633, 1646, 1667, 1741, 1832, 1925,
               ],
    "bucket": [100431, 100435, 100438, 100439, 100441, 100444, 100446, 100448, 100454, 100461,
               100462, 100468, 100470, 100473, 100482, 100484, 100486, 102352, 102358,
               ],
    "laptop": [9748, 9912, 9918, 9960, 9968, 9992, 11395, 11405, 11406, 11477, 11581, 11586, 9996, 10090, 10098, 10101, 10125],
    "toilet": [101320, 102621, 102622, 102630, 102634, 102645, 102648, 102651, 102652, 102654,
                102658, 102677, 102687, 102689, 102692, 102697, 102699, 102701, 102703, 102707,
               102708, 103234, 102663, 102666, 102667, 102669, 102670, 102675],
}

# Camera config
CAMERA_CONFIG = {
    "faucet": {
        "instance_1": dict(position=np.array([-0.3, 0.6, 0.4]), look_at_dir=np.array([0.16, -0.7, -0.35]),
                           # from left side.
                           right_dir=np.array([-1.5, -2, 0]), fov=np.deg2rad(69.4), resolution=(84, 84)),
    },
    "bucket": {
        "instance_1": dict(
            pose=sapien.Pose(p=np.array([0, 1, 0.5]), q=transforms3d.euler.euler2quat(np.pi / 2, np.pi, 0)),
            fov=np.deg2rad(69.4), resolution=(84, 84)),
    },
    "laptop": {
        "instance_1": dict(
            pose=sapien.Pose(p=np.array([0, 1, 0.5]), q=transforms3d.euler.euler2quat(np.pi / 2, np.pi, 0)),
            fov=np.deg2rad(69.4), resolution=(84, 84)),
    },
    "toilet": {
        "instance_1": dict(
            pose=sapien.Pose(p=np.array([0, 1, 0.5]), q=transforms3d.euler.euler2quat(np.pi / 2, np.pi, 0)),
            fov=np.deg2rad(69.4), resolution=(84, 84)),
    },
    
    # "faucet": {
    #     "instance_1": dict(position=np.array([-0.3, 0.6, 0.4]), look_at_dir=np.array([0.16, -0.7, -0.35]),
    #                        # from left side.
    #                        right_dir=np.array([-1.5, -2, 0]), fov=np.deg2rad(69.4), resolution=(1024, 1024)),
    # },
    # "bucket": {
    #     "instance_1": dict(
    #         pose=sapien.Pose(p=np.array([0, 1, 0.5]), q=transforms3d.euler.euler2quat(np.pi / 2, np.pi, 0)),
    #         fov=np.deg2rad(69.4), resolution=(1024, 1024)),
    # },
    # "laptop": {
    #     "instance_1": dict(
    #         pose=sapien.Pose(p=np.array([0, 1, 0.5]), q=transforms3d.euler.euler2quat(np.pi / 2, np.pi, 0)),
    #         fov=np.deg2rad(69.4), resolution=(1024, 1024)),
    # },
    # "toilet": {
    #     "instance_1": dict(
    #         pose=sapien.Pose(p=np.array([0, 1, 0.5]), q=transforms3d.euler.euler2quat(np.pi / 2, np.pi, 0)),
    #         fov=np.deg2rad(69.4), resolution=(1024, 1024)),
    # },
    
    
    "viz_only": {  # only for visualization (human), not for visual observation
        "faucet_viz": dict(position=np.array([-0.3, 0.6, 0.4]), look_at_dir=np.array([0.16, -0.7, -0.35]),
                           # from left side.
                           right_dir=np.array([-1.5, -2, 0]), fov=np.deg2rad(69.4), resolution=(1000, 1000)),
        "faucet_viz2": dict(
            pose=sapien.Pose(p=np.array([0, 0.8, 0.5]), q=transforms3d.euler.euler2quat(np.pi / 3, np.pi, 0)),
            fov=np.deg2rad(69.4), resolution=(1000, 1000)),
        "bucket_viz": dict(
            pose=sapien.Pose(p=np.array([0, 1, 0.5]), q=transforms3d.euler.euler2quat(np.pi / 2, np.pi, 0)),
            fov=np.deg2rad(69.4), resolution=(1000, 1000)),
        "laptop_viz": dict(
            pose=sapien.Pose(p=np.array([0, 1, 0.5]), q=transforms3d.euler.euler2quat(np.pi / 2, np.pi, 0)),
            fov=np.deg2rad(69.4), resolution=(1000, 1000)),
        "toilet_viz": dict(
            pose=sapien.Pose(p=np.array([0, 1, 0.5]), q=transforms3d.euler.euler2quat(np.pi / 2, np.pi, 0)),
            fov=np.deg2rad(69.4), resolution=(1000, 1000)),
    },
}

EVAL_CAM_NAMES_CONFIG = {
    "faucet": ["faucet_viz"],
    "bucket": ['bucket_viz'],
    "laptop": ['laptop_viz'],
    "toilet": ['toilet_viz'],
}

# Observation config type
OBS_CONFIG = {
    "instance_rgb": {
        "instance_1": {"rgb": True, "depth":True},
    },
    "instance": {
        "instance_1": {"point_cloud": {"num_points": 1024}, "rgb": True, "depth":True},
    },
    "instance_noise": {
        "instance_1": {
            "point_cloud": {"num_points": 1024, "pose_perturb_level": 0.5,
                            "process_fn_kwargs": {"noise_level": 0.5}},
            "rgb": True,
            "depth":True
        },
    },
    "instance_pc_seg": {
        "instance_1": {
            "point_cloud": {"use_seg": True, "use_2frame": True, "num_points": 1024, "pose_perturb_level": 0.5,
                            "process_fn_kwargs": {"noise_level": 0.5}},
            "rgb": True,
            "depth":True
        },
    },
}

# Imagination config type
IMG_CONFIG = {
    "robot": {
        "robot": {
            # "link_base": 8, "link1": 8, "link2": 8, "link3": 8, "link4": 8, "link5": 8, "link6": 8,
            "link_15.0_tip": 8, "link_3.0_tip": 8, "link_7.0_tip": 8, "link_11.0_tip": 8,
            "link_15.0": 8, "link_3.0": 8, "link_7.0": 8, "link_11.0": 8,
            "link_14.0": 8, "link_2.0": 8, "link_6.0": 8, "link_10.0": 8,  # "base_link": 8
        },
    }
}

RANDOM_CONFIG = {"bucket": {"rand_pos": 0.05, "rand_degree": 0}, "laptop": {"rand_pos": 0.1, "rand_degree": 60},
                 "faucet": {"rand_pos": 0.1, "rand_degree": 90}, 'toilet': {"rand_pos": 0.2, "rand_degree": 45},}
