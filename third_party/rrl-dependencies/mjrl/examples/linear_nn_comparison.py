from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.policies.gaussian_linear import LinearPolicy
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.utils.train_agent import train_agent
import mjrl.envs
import time as timer
SEED = 500

# NN policy
# ==================================
e = GymEnv('mjrl_swimmer-v0')
policy = MLP(e.spec, hidden_sizes=(32,32), seed=SEED)
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
agent = NPG(e, policy, baseline, normalized_step_size=0.1, seed=SEED, save_logs=True)

ts = timer.time()
train_agent(job_name='swimmer_nn_exp1',
            agent=agent,
            seed=SEED,
            niter=50,
            gamma=0.995,  
            gae_lambda=0.97,
            num_cpu=1,
            sample_mode='trajectories',
            num_traj=10,
            save_freq=5,
            evaluation_rollouts=5)
print("time taken for NN policy training = %f" % (timer.time()-ts))


# Linear policy
# ==================================
e = GymEnv('mjrl_swimmer-v0')
policy = LinearPolicy(e.spec, seed=SEED)
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3)
agent = NPG(e, policy, baseline, normalized_step_size=0.1, seed=SEED, save_logs=True)

ts = timer.time()
train_agent(job_name='swimmer_linear_exp1',
            agent=agent,
            seed=SEED,
            niter=50,
            gamma=0.995,  
            gae_lambda=0.97,
            num_cpu=1,
            sample_mode='trajectories',
            num_traj=10,
            save_freq=5,
            evaluation_rollouts=5)
print("time taken for linear policy training = %f" % (timer.time()-ts))
