from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.behavior_cloning import BC
from mjrl.utils.train_agent import train_agent
from mjrl.samplers.core import sample_paths
import mjrl.envs
import time as timer
import pickle
SEED = 500

# ------------------------------
# Train expert policy first
e = GymEnv('mjrl_swimmer-v0')
policy = MLP(e.spec, hidden_sizes=(32,32), seed=SEED)
baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, epochs=5, learn_rate=1e-3)
agent = NPG(e, policy, baseline, normalized_step_size=0.1, seed=SEED, save_logs=True)

ts = timer.time()
print("========================================")
print("Training expert policy")
print("========================================")
train_agent(job_name='swimmer_exp1',
            agent=agent,
            seed=SEED,
            niter=50,
            gamma=0.995,
            gae_lambda=0.97,
            num_cpu=1,
            sample_mode='trajectories',
            num_traj=10,
            save_freq=5,
            evaluation_rollouts=None)
print("========================================")
print("Expert policy training complete !!!")
print("========================================")
print("time taken = %f" % (timer.time()-ts))
print("========================================")

# ------------------------------
# Get demonstrations
print("========================================")
print("Collecting expert demonstrations")
print("========================================")
expert_pol = pickle.load(open('swimmer_exp1/iterations/best_policy.pickle', 'rb'))
demo_paths = sample_paths(num_traj=5, policy=expert_pol, env=e.env_id)

# ------------------------------
# Train BC
policy = MLP(e.spec, hidden_sizes=(32,32), seed=SEED)
bc_agent = BC(demo_paths, policy=policy, epochs=20, batch_size=64, lr=1e-3) # will use Adam by default
ts = timer.time()
print("========================================")
print("Running BC with expert demonstrations")
print("========================================")
bc_agent.train()
print("========================================")
print("BC training complete !!!")
print("time taken = %f" % (timer.time()-ts))
print("========================================")

# ------------------------------
# Evaluate Policies
bc_pol_score = e.evaluate_policy(policy, num_episodes=5, mean_action=True)
expert_score = e.evaluate_policy(expert_pol, num_episodes=5, mean_action=True)
print("Expert policy performance (eval mode) = %f" % expert_score[0][0])
print("BC policy performance (eval mode) = %f" % bc_pol_score[0][0])
