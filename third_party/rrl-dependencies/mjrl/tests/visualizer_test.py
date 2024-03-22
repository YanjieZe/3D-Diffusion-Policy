from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.utils.train_agent import train_agent
import mjrl.envs
import time as timer
SEED = 500

e = GymEnv('mjrl_point_mass-v0')
policy = MLP(e.spec, hidden_sizes=(32,32), seed=SEED)
baseline = QuadraticBaseline(e.spec)
agent = NPG(e, policy, baseline, normalized_step_size=0.5, seed=SEED, save_logs=True)

ts = timer.time()
train_agent(job_name='vis_exp',
            agent=agent,
            seed=SEED,
            niter=10,
            gamma=0.95,
            gae_lambda=0.97,
            num_cpu=1,
            sample_mode='trajectories',
            num_traj=100,
            save_freq=5,
            evaluation_rollouts=None)
print("time taken = %f" % (timer.time()-ts))
e.visualize_policy(policy, num_episodes=5, horizon=e.horizon, mode='exploration')
