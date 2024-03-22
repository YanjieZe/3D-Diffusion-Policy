import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import zarr
import torch
import numpy as np
import torch.nn.functional as F
import pytorch3d.ops as torch3d_ops
from dexart.env.task_setting import TRAIN_CONFIG, RANDOM_CONFIG
from dexart.env.create_env import create_env
from stable_baselines3 import PPO
# from examples.train import get_3d_policy_kwargs
from train import get_3d_policy_kwargs
from tqdm import tqdm
from termcolor import cprint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=10, help='number of total episodes')
    parser.add_argument('--use_test_set', dest='use_test_set', action='store_true', default=False)
    parser.add_argument('--root_dir', type=str, default='data', help='directory to save data')
    parser.add_argument('--img_size', type=int, default=84, help='image size')
    parser.add_argument('--num_points', type=int, default=1024, help='number of points in point cloud')
    args = parser.parse_args()
    return args

def downsample_with_fps(points: np.ndarray, num_points: int = 512):
    # fast point cloud sampling using torch3d
    points = torch.from_numpy(points).unsqueeze(0).cuda()
    num_points = torch.tensor([num_points]).cuda()
    # remember to only use coord to sample
    _, sampled_indices = torch3d_ops.sample_farthest_points(points=points[...,:3], K=num_points)
    points = points.squeeze(0).cpu().numpy()
    points = points[sampled_indices.squeeze(0).cpu().numpy()]
    return points

def main():
    args = parse_args()
    task_name = args.task_name
    use_test_set = args.use_test_set
    checkpoint_path = args.checkpoint_path
    

    save_dir = os.path.join(args.root_dir, 'dexart_'+args.task_name+'_expert.zarr')
    if os.path.exists(save_dir):
        cprint('Data already exists at {}'.format(save_dir), 'red')
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        # user_input = input()
        user_input = 'y'
        if user_input == 'y':
            cprint('Overwriting {}'.format(save_dir), 'red')
            os.system('rm -rf {}'.format(save_dir))
        else:
            cprint('Exiting', 'red')
            return
    os.makedirs(save_dir, exist_ok=True)
    

    if use_test_set:
        indeces = TRAIN_CONFIG[task_name]['unseen']
        cprint(f"using unseen instances {indeces}", 'yellow')
    else:
        indeces = TRAIN_CONFIG[task_name]['seen']
        cprint(f"using seen instances {indeces}", 'yellow')

    rand_pos = RANDOM_CONFIG[task_name]['rand_pos']
    rand_degree = RANDOM_CONFIG[task_name]['rand_degree']
    env = create_env(task_name=task_name,
                     use_visual_obs=True,
                     use_gui=False,
                     is_eval=True,
                     pc_noise=True,
                     pc_seg=True,
                     index=indeces,
                     img_type='robot',
                     rand_pos=rand_pos,
                     rand_degree=rand_degree)

    policy = PPO.load(checkpoint_path, env, 'cuda:0',
                      policy_kwargs=get_3d_policy_kwargs(extractor_name='smallpn'),
                      check_obs_space=False, force_load=True)

    eval_instances = len(env.instance_list)
    num_episodes = args.num_episodes
    cprint(f"generate {num_episodes} episodes in total", 'yellow')
    
    success_list = []
    reward_list = []
    
    total_count = 0
    img_arrays = []
    point_cloud_arrays = []
    depth_arrays = []
    state_arrays = []
    imagin_robot_arrays = []
    action_arrays = []
    episode_ends_arrays = []        


    with tqdm(total=num_episodes) as pbar:
        num_success = 0
        while num_success < num_episodes:

            # obs dict keys: 'instance_1-seg_gt', 'instance_1-point_cloud', 
            # 'instance_1-rgb', 'imagination_robot', 'state', 'oracle_state'
            obs = env.reset() 
            eval_success = False
            reward_sum = 0
            
            img_arrays_sub = []
            point_cloud_arrays_sub = []
            depth_arrays_sub = []
            state_arrays_sub = []
            imagin_robot_arrays_sub = []
            action_arrays_sub = []
            total_count_sub = 0
            for j in range(env.horizon):
                
                if isinstance(obs, dict):
                    for key, value in obs.items():
                        obs[key] = value[np.newaxis, :]
                else:
                    obs = obs[np.newaxis, :]
                action = policy.predict(observation=obs, deterministic=True)[0]
                
                # fetch data
                total_count_sub += 1
                obs_state = obs['state'][0] # (32)
                obs_imagin_robot = obs['imagination_robot'][0] # (96,7)
                obs_point_cloud = obs['instance_1-point_cloud'][0] # (1024,3)
                obs_depth = obs['instance_1-depth'][0] # (84,84)
                
                if obs_point_cloud.shape[0] > args.num_points:
                    obs_point_cloud = downsample_with_fps(obs_point_cloud, num_points=args.num_points)
                obs_image = obs['instance_1-rgb'][0] # (84,84,3), [0,1]
                

                # to 0-255
                obs_image = (obs_image*255).astype(np.uint8)
                
                # interpolate to target image size
                if obs_image.shape[0] != args.img_size:
                    obs_image = F.interpolate(torch.from_numpy(obs_image).permute(2,0,1).unsqueeze(0), 
                                            size=args.img_size).squeeze().permute(1,2,0).numpy()
                # save data
                img_arrays_sub.append(obs_image)
                imagin_robot_arrays_sub.append(obs_imagin_robot)
                point_cloud_arrays_sub.append(obs_point_cloud)
                depth_arrays_sub.append(obs_depth)
                state_arrays_sub.append(obs_state)
                action_arrays_sub.append(action)
                
                # step
                obs, reward, done, _ = env.step(action)
                reward_sum += reward
                if env.is_eval_done:
                    eval_success = True
                if done:
                    break
            
            if eval_success:
                total_count += total_count_sub
                episode_ends_arrays.append(total_count) # the index of the last step of the episode    
                reward_list.append(reward_sum)
                success_list.append(int(eval_success))
                
                img_arrays.extend(img_arrays_sub)
                imagin_robot_arrays.extend(imagin_robot_arrays_sub)
                point_cloud_arrays.extend(point_cloud_arrays_sub)
                depth_arrays.extend(depth_arrays_sub)
                state_arrays.extend(state_arrays_sub)
                action_arrays.extend(action_arrays_sub)
                
                num_success += 1
                
                pbar.update(1)
                pbar.set_description(f"reward = {reward_sum}, success = {eval_success}")
            else:
                print("episode failed. continue.")
                continue
                    
                
    cprint(f"reward_mean = {np.mean(reward_list)}, success rate = {np.mean(success_list)}", 'yellow')
    
    ###############################
    # save data
    ###############################
    # create zarr file
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save img, state, action arrays into data, and episode ends arrays into meta
    img_arrays = np.stack(img_arrays, axis=0)
    if img_arrays.shape[1] == 3: # make channel last
        img_arrays = np.transpose(img_arrays, (0,2,3,1))
    state_arrays = np.stack(state_arrays, axis=0)
    imagin_robot_arrays = np.stack(imagin_robot_arrays, axis=0)
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    depth_arrays = np.stack(depth_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    img_chunk_size = (env.horizon, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
    imagin_robot_chunk_size = (env.horizon, imagin_robot_arrays.shape[1], imagin_robot_arrays.shape[2])
    point_cloud_chunk_size = (env.horizon, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    depth_chunk_size = (env.horizon, depth_arrays.shape[1], depth_arrays.shape[2])
    state_chunk_size = (env.horizon, state_arrays.shape[1])
    action_chunk_size = (env.horizon, action_arrays.shape[1])
    zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('imagin_robot', data=imagin_robot_arrays, chunks=imagin_robot_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
    
    # print shape
    cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
    cprint(f'imagin_robot shape: {imagin_robot_arrays.shape}, range: [{np.min(imagin_robot_arrays)}, {np.max(imagin_robot_arrays)}]', 'green')
    cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
    cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
    cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')
    

if __name__ == "__main__":
    main()
    
    
