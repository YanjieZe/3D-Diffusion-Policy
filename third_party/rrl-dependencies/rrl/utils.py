# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur
# Copyright (c) Facebook, Inc. and its affiliates

from mjrl.utils.gym_env import GymEnv
from rrl.multicam import RRL
import os

def make_env(env, cam_list=[], from_pixels=False, encoder_type=None, hybrid_state=None) :
    e = GymEnv(env)
    env_kwargs = None
    if from_pixels :
        height = 84
        width = 84
        latent_dim = height*width*len(cam_list)*3

    if encoder_type and encoder_type == 'resnet34':
        assert from_pixels==True
        height = 256
        width = 256
        latent_dim = 512*len(cam_list)
    if from_pixels:
        e = RRL(e, cameras=cam_list, encoder_type=encoder_type,
            height=height, width=width, latent_dim=latent_dim, hybrid_state=hybrid_state)
        env_kwargs = {'rrl_kwargs' : e.env_kwargs}
    return e, env_kwargs

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def preprocess_args(args):
	job_data = {}
	job_data['seed'] = args.seed
	job_data['env'] = args.env
	job_data['output'] = args.output
	job_data['from_pixels'] = args.from_pixels
	job_data['hybrid_state'] = args.hybrid_state
	job_data['stack_frames'] = args.stack_frames
	job_data['encoder_type'] = args.encoder_type
	job_data['cam1'] = args.cam1
	job_data['cam2'] = args.cam2
	job_data['cam3'] = args.cam3
	job_data['algorithm'] = args.algorithm
	job_data['num_cpu'] = args.num_cpu
	job_data['save_freq'] = args.save_freq
	job_data['eval_rollouts'] = args.eval_rollouts
	job_data['demo_file'] = args.demo_file
	job_data['bc_batch_size'] = args.bc_batch_size
	job_data['bc_epochs'] = args.bc_epochs
	job_data['bc_learn_rate'] = args.bc_learn_rate
	#job_data['policy_size'] = args.policy_size
	job_data['policy_size'] = tuple(map(int, args.policy_size.split(', ')))
	job_data['vf_batch_size'] = args.vf_batch_size
	job_data['vf_epochs'] = args.vf_epochs
	job_data['vf_learn_rate'] = args.vf_learn_rate
	job_data['rl_step_size'] = args.rl_step_size
	job_data['rl_gamma'] = args.rl_gamma
	job_data['rl_gae'] = args.rl_gae
	job_data['rl_num_traj'] = args.rl_num_traj
	job_data['rl_num_iter'] = args.rl_num_iter
	job_data['lam_0'] = args.lam_0
	job_data['lam_1'] = args.lam_1
	print(job_data)
	return job_data


