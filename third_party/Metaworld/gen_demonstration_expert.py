# bash scripts/metaworld/gen_demonstration_expert.sh reach 5
import argparse
import os
import zarr
import numpy as np
from diffusion_policy_3d.env import MetaWorldEnv
from termcolor import cprint
import copy
import imageio
from metaworld.policies import *
# import faulthandler
# faulthandler.enable()

seed = np.random.randint(0, 100)

def load_mw_policy(task_name):
	if task_name == 'peg-insert-side':
		agent = SawyerPegInsertionSideV2Policy()
	else:
		task_name = task_name.split('-')
		task_name = [s.capitalize() for s in task_name]
		task_name = "Sawyer" + "".join(task_name) + "V2Policy"
		agent = eval(task_name)()
	return agent

def main(args):
	env_name = args.env_name

	
	save_dir = os.path.join(args.root_dir, 'metaworld_'+args.env_name+'_expert.zarr')
	if os.path.exists(save_dir):
		cprint('Data already exists at {}'.format(save_dir), 'red')
		cprint("If you want to overwrite, delete the existing directory first.", "red")
		cprint("Do you want to overwrite? (y/n)", "red")
		user_input = 'y'
		if user_input == 'y':
			cprint('Overwriting {}'.format(save_dir), 'red')
			os.system('rm -rf {}'.format(save_dir))
		else:
			cprint('Exiting', 'red')
			return
	os.makedirs(save_dir, exist_ok=True)

	e = MetaWorldEnv(env_name, device="cuda:0", use_point_crop=True)
	
	num_episodes = args.num_episodes
	cprint(f"Number of episodes : {num_episodes}", "yellow")
	

	total_count = 0
	img_arrays = []
	point_cloud_arrays = []
	depth_arrays = []
	state_arrays = []
	full_state_arrays = []
	action_arrays = []
	episode_ends_arrays = []
    
	
	episode_idx = 0
	

	mw_policy = load_mw_policy(env_name)
	
	# loop over episodes
	while episode_idx < num_episodes:
		raw_state = e.reset()['full_state']

		obs_dict = e.get_visual_obs()

		
		done = False
		
		ep_reward = 0.
		ep_success = False
		ep_success_times = 0
		

		img_arrays_sub = []
		point_cloud_arrays_sub = []
		depth_arrays_sub = []
		state_arrays_sub = []
		full_state_arrays_sub = []
		action_arrays_sub = []
		total_count_sub = 0
  
		while not done:

			total_count_sub += 1
			
			obs_img = obs_dict['image']
			obs_robot_state = obs_dict['agent_pos']
			obs_point_cloud = obs_dict['point_cloud']
			obs_depth = obs_dict['depth']
   

			img_arrays_sub.append(obs_img)
			point_cloud_arrays_sub.append(obs_point_cloud)
			depth_arrays_sub.append(obs_depth)
			state_arrays_sub.append(obs_robot_state)
			full_state_arrays_sub.append(raw_state)
			
			action = mw_policy.get_action(raw_state)
		
			action_arrays_sub.append(action)
			obs_dict, reward, done, info = e.step(action)
			raw_state = obs_dict['full_state']
			ep_reward += reward
   

			ep_success = ep_success or info['success']
			ep_success_times += info['success']
   
			if done:
				break
		

		if not ep_success or ep_success_times < 5:
			cprint(f'Episode: {episode_idx} failed with reward {ep_reward} and success times {ep_success_times}', 'red')
			continue
		else:
			total_count += total_count_sub
			episode_ends_arrays.append(copy.deepcopy(total_count)) # the index of the last step of the episode    
			img_arrays.extend(copy.deepcopy(img_arrays_sub))
			point_cloud_arrays.extend(copy.deepcopy(point_cloud_arrays_sub))
			depth_arrays.extend(copy.deepcopy(depth_arrays_sub))
			state_arrays.extend(copy.deepcopy(state_arrays_sub))
			action_arrays.extend(copy.deepcopy(action_arrays_sub))
			full_state_arrays.extend(copy.deepcopy(full_state_arrays_sub))
			cprint('Episode: {}, Reward: {}, Success Times: {}'.format(episode_idx, ep_reward, ep_success_times), 'green')
			episode_idx += 1
	

	# save data
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
	full_state_arrays = np.stack(full_state_arrays, axis=0)
	point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
	depth_arrays = np.stack(depth_arrays, axis=0)
	action_arrays = np.stack(action_arrays, axis=0)
	episode_ends_arrays = np.array(episode_ends_arrays)

	compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
	img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
	state_chunk_size = (100, state_arrays.shape[1])
	full_state_chunk_size = (100, full_state_arrays.shape[1])
	point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
	depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
	action_chunk_size = (100, action_arrays.shape[1])
	zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('full_state', data=full_state_arrays, chunks=full_state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
	zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

	cprint(f'-'*50, 'cyan')
	# print shape
	cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
	cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
	cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
	cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
	cprint(f'full_state shape: {full_state_arrays.shape}, range: [{np.min(full_state_arrays)}, {np.max(full_state_arrays)}]', 'green')
	cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
	cprint(f'Saved zarr file to {save_dir}', 'green')

	# clean up
	del img_arrays, state_arrays, point_cloud_arrays, action_arrays, episode_ends_arrays
	del zarr_root, zarr_data, zarr_meta
	del e


 
if __name__ == "__main__":
    
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='basketball')
	parser.add_argument('--num_episodes', type=int, default=10)
	parser.add_argument('--root_dir', type=str, default="../../3D-Diffusion-Policy/data/" )

	args = parser.parse_args()
	main(args)
