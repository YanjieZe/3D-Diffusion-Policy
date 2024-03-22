import sapien.core as sapien
from typing import Dict

import numpy as np


def fetch_texture(cam: sapien.CameraEntity, texture_name: str, return_torch=False):
    dlpack = cam.get_dl_tensor(texture_name)
    if not return_torch:
        assert texture_name not in ["Segmentation"]
        shape = sapien.dlpack.dl_shape(dlpack)
        output_array = np.zeros(shape, dtype=np.float32)
        sapien.dlpack.dl_to_numpy_cuda_async_unchecked(dlpack, output_array)
        sapien.dlpack.dl_cuda_sync()
        return output_array
    else:
        import torch
        return torch.from_dlpack(dlpack)


def generate_imagination_pc_from_obs(obs: Dict[str, np.ndarray]):
    # has_pc = False
    pc = []
    color = []
    category = []
    for key, value in obs.items():
        if "point_cloud" in key:
            # has_pc = True
            num_points = value.shape[0]
            pc.append(value[:, :])
            color.append(np.tile(np.array([0, 0, 255]), [num_points, 1]))
            category.append([0] * num_points)
        elif key == "imagination_robot":
            num_points = value.shape[0]
            pc.append(value[:, :3])
            color.append(np.tile(np.array([255, 0, 0]), [num_points, 1]))
            category.append([1] * num_points)
        elif key == "imagination_instance":
            num_points = value.shape[0]
            pc.append(value[:, :3])
            color.append(np.tile(np.array([0, 128, 128]), [num_points, 1]))
            category.append([3] * num_points)
        elif 'pc_seg' in key:  # e.g.  key: 'faucet_1-pc_seg'
            group_num = value.shape[1] - 3  # 7-3=4
            assert group_num == 4
            colors = [  # might not have been used.
                np.array([0, 0, 255]),
                np.array([255, 0, 0]),
                np.array([0, 255, 0]),
                np.array([0, 128, 128])
            ]
            for i in range(group_num):
                seg_idx = np.argmax(value[..., 3:], axis=1)
                idx = np.where(seg_idx == i)[0]
                value_i = value[idx, :3]
                num_points = value_i.shape[0]
                if num_points == 0:
                    continue
                pc.append(value_i)
                color.append(np.tile(colors[i], [num_points, 1]))
                category.append([i] * num_points)

    # if has_pc:
    pc = np.concatenate(pc)
    color = np.concatenate(color).astype(np.uint8)
    category = np.concatenate(category)[:, None]
    return pc, color, category
    # else:
    #     return None, None, None
