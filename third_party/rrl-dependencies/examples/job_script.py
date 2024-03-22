# Copyright (c) Rutav Shah, Indian Institute of Technlogy Kharagpur
# Copyright (c) Facebook, Inc. and its affiliates

# python job_script.py --config dapg.txt --output dir_name --cam1 cam_name --cam2 cam_name --cam3 cam_name
"""
This is a job script for running NPG/DAPG on hand tasks and other gym envs.
Note that DAPG generalizes PG and BC init + PG finetuning.
With appropriate settings of parameters, we can recover the full family.
"""

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.dapg import DAPG
from mjrl.algos.behavior_cloning import BC
from mjrl.utils.train_agent import train_agent
from mjrl.samplers.core import sample_paths
import os
import json
import mjrl.envs
import mj_envs
import time as timer
import pickle
import argparse
from pathlib import Path
import multiprocessing
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from rrl.utils import make_env, preprocess_args
home = str(Path.home())

# ===============================================================================
# Get command line arguments
# ===============================================================================

@hydra.main(config_name="hammer_dapg", config_path="config")
def main(args : DictConfig):
    job_data = preprocess_args(args)
    #with open(args.config, 'r') as f:
    #	job_data = eval(f.read())
    assert 'algorithm' in job_data.keys()
    assert any([job_data['algorithm'] == a for a in ['NPG', 'BCRL', 'DAPG']])
    job_data['lam_0'] = 0.0 if 'lam_0' not in job_data.keys() else job_data['lam_0']
    job_data['lam_1'] = 0.0 if 'lam_1' not in job_data.keys() else job_data['lam_1']

    #os.mkdir(args.output)
    JOB_DIR = job_data['output'] + "_seed{}".format(job_data['seed'])
    if not os.path.exists(JOB_DIR):
        os.mkdir(JOB_DIR)
    EXP_FILE = JOB_DIR + '/job_config.json'
    with open(EXP_FILE, 'w') as f:
        json.dump(job_data, f, indent=4)

    # ===============================================================================
    # Train Loop
    # ===============================================================================

    if job_data['from_pixels'] == True :
        if args.cam1 is None:
            print("Please pass cameras in the arguments.")
            exit()

        encoder = None
        cam_list = [args.cam1] # Change this behavior. Pass list in hydra configs
        if args.cam2 is not None:
        	cam_list.append(args.cam2)
        	if args.cam3 is not None:
        		cam_list.append(args.cam3)

        num_cam = len(cam_list)
        camera_type = cam_list[0]
        if num_cam > 1:
            camera_type = "multicam"
        e, env_kwargs = make_env(job_data['env'], from_pixels=True, cam_list=cam_list, encoder_type=job_data['encoder_type'], hybrid_state=job_data['hybrid_state'])
    else :
        e, env_kwargs = make_env(job_data['env'])

    policy = MLP(e.spec, hidden_sizes=job_data['policy_size'], seed=job_data['seed'])
    baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=job_data['vf_batch_size'],
                           epochs=job_data['vf_epochs'], learn_rate=job_data['vf_learn_rate'])
    # Get demonstration data if necessary and behavior clone
    if job_data['algorithm'] != 'NPG':
        print("========================================")
        print("Collecting expert demonstrations")
        print("========================================")
        demo_paths = pickle.load(open(job_data['demo_file'], 'rb'))
        print("Number of demo paths : ", len(demo_paths))

        bc_agent = BC(demo_paths, policy=policy, epochs=job_data['bc_epochs'], batch_size=job_data['bc_batch_size'],
                      lr=job_data['bc_learn_rate'], loss_type='MSE', set_transforms=False)
        in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
        bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
        bc_agent.set_variance_with_data(out_scale)
        ts = timer.time()
        print("========================================")
        print("Running BC with expert demonstrations")
        print("========================================")
        bc_agent.train()
        print("========================================")
        print("BC training complete !!!")
        print("time taken = %f" % (timer.time() - ts))
        print("========================================")

        if job_data['eval_rollouts'] >= 1:
            score = e.evaluate_policy(policy, num_episodes=job_data['eval_rollouts'], mean_action=True)
            print("Score with behavior cloning = %f" % score[0][0])
            try :
                print("Number of cpu: ", job_data['num_cpu'])
                eval_paths = sample_paths(num_traj=job_data['eval_rollouts'], policy=policy, num_cpu=job_data["num_cpu"],
                                          	env=e.env_id, eval_mode=True, base_seed=job_data["seed"], env_kwargs=env_kwargs)
                success_rate = e.env.env.evaluate_success(eval_paths)
                print("Success Rate :", success_rate)
            except :
            	pass
            pickle.dump(policy, open(JOB_DIR + '/demo_bs{}_epochs{}.pickle'.format(job_data['bc_batch_size'], job_data['bc_epochs']), 'wb'))
    if job_data['algorithm'] != 'DAPG':
        # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
        demo_paths = None

    # ===============================================================================
    # RL Loop
    # ===============================================================================
    # policy.log_std_val *= 3.
    rl_agent = DAPG(e, policy, baseline, demo_paths,
                    normalized_step_size=job_data['rl_step_size'],
                    lam_0=job_data['lam_0'], lam_1=job_data['lam_1'],
                    seed=job_data['seed'], save_logs=True
                    )

    print("========================================")
    print("Starting reinforcement learning phase")
    print("========================================")

    ts = timer.time()
    train_agent(job_name=JOB_DIR,
                agent=rl_agent,
                seed=job_data['seed'],
                niter=job_data['rl_num_iter'],
                gamma=job_data['rl_gamma'],
                gae_lambda=job_data['rl_gae'],
                num_cpu=job_data['num_cpu'],
                sample_mode='trajectories',
                num_traj=job_data['rl_num_traj'],
                save_freq=job_data['save_freq'],
                evaluation_rollouts=job_data['eval_rollouts'],
                env_kwargs=env_kwargs)
    print("time taken = %f" % (timer.time()-ts))


if __name__ == '__main__' :
	multiprocessing.set_start_method('spawn')
	main()
