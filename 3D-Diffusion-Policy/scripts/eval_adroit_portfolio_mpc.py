import argparse
import copy
import json
import os
import pathlib
import sys
from collections import deque

import dill
import numpy as np
import torch

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
os.chdir(str(ROOT_DIR))

from train import TrainDP3Workspace  # noqa: E402
from diffusion_policy_3d.common.pytorch_util import dict_apply  # noqa: E402
from diffusion_policy_3d.env import AdroitEnv  # noqa: E402
from diffusion_policy_3d.gym_util.mjpc_diffusion_wrapper import (  # noqa: E402
    MujocoPointcloudWrapperAdroit,
)
from scripts.collect_adroit_branch_data import goal_value, stack_obs  # noqa: E402


def load_policy(checkpoint, device):
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    workspace = TrainDP3Workspace(payload["cfg"])
    workspace.load_payload(payload)
    policy = workspace.ema_model if workspace.cfg.training.use_ema else workspace.model
    policy.execution_mode = "fixed"
    policy.eval()
    policy.to(device)
    return policy


def parse_candidate(spec):
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(
            "candidate must be name:checkpoint:chunk, "
            f"got {spec}")
    name, checkpoint, chunk = parts
    return name, checkpoint, int(chunk)


def score_rollout(env, state, actions, chunk, task):
    env.set_env_state(copy.deepcopy(state))
    reward_sum = 0.0
    goal_sum = 0.0
    final_goal = 0.0
    done = False
    for act in actions[:chunk]:
        _, reward, done, info = env.step(act)
        reward_sum += float(reward)
        final_goal = goal_value(info)
        goal_sum += final_goal
        if done:
            break
    success_bonus = 100.0 if final_goal > 0 else 0.0
    if task == "door":
        return success_bonus + 25.0 * final_goal + 3.0 * goal_sum + 5.0 * reward_sum
    return success_bonus + 20.0 * final_goal + 2.0 * goal_sum + 5.0 * reward_sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["door", "hammer"], required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_point_crop", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    specs = [parse_candidate(x) for x in args.candidate]
    policies = []
    for name, checkpoint, chunk in specs:
        policies.append({
            "name": name,
            "checkpoint": checkpoint,
            "chunk": chunk,
            "policy": load_policy(checkpoint, device),
        })

    env = MujocoPointcloudWrapperAdroit(
        env=AdroitEnv(env_name=args.task, use_point_cloud=True),
        env_name=f"adroit_{args.task}",
        use_point_crop=not args.no_point_crop,
    )
    n_obs_steps = int(policies[0]["policy"].n_obs_steps)

    successes = []
    goals = []
    calls = []
    chosen_names = []
    chosen_chunks = []

    for episode in range(args.episodes):
        obs = env.reset()
        history = deque([obs], maxlen=n_obs_steps)
        for candidate in policies:
            candidate["policy"].reset()
        done = False
        step_count = 0
        episode_goal = 0.0
        episode_calls = 0
        last_info = {}

        while not done and step_count < args.max_steps:
            stacked = stack_obs(history, n_obs_steps)
            obs_t = dict_apply(
                stacked,
                lambda x: torch.from_numpy(x).to(device=device).unsqueeze(0),
            )
            obs_input = {
                "point_cloud": obs_t["point_cloud"],
                "agent_pos": obs_t["agent_pos"],
            }
            state = env.get_env_state()
            proposals = []
            for candidate in policies:
                with torch.no_grad():
                    action_dict = candidate["policy"].predict_action(obs_input)
                action_pred = action_dict["action_pred"].detach().cpu().numpy().squeeze(0)
                start = n_obs_steps - 1
                chunk = min(candidate["chunk"], action_pred.shape[0] - start)
                plan = action_pred[start:start + chunk]
                score = score_rollout(env, state, plan, chunk, args.task)
                proposals.append((score, candidate["name"], chunk, plan))
            env.set_env_state(copy.deepcopy(state))
            score, name, chunk, plan = max(proposals, key=lambda x: x[0])
            chosen_names.append(name)
            chosen_chunks.append(chunk)
            episode_calls += 1

            for act in plan[:chunk]:
                obs, _, done, last_info = env.step(act)
                history.append(obs)
                episode_goal += goal_value(last_info)
                step_count += 1
                if done or step_count >= args.max_steps:
                    break

        success = float(goal_value(last_info) > 0)
        successes.append(success)
        goals.append(episode_goal)
        calls.append(episode_calls)
        print(
            f"episode={episode:03d} success={success:.0f} "
            f"goal={episode_goal:.1f} calls={episode_calls}",
            flush=True,
        )

    choice_counts = {
        name: int(sum(1 for x in chosen_names if x == name))
        for name, _, _ in specs
    }
    result = {
        "task": args.task,
        "execution_mode": "portfolio_mpc",
        "episodes": args.episodes,
        "mean_success_rates": float(np.mean(successes)),
        "mean_n_goal_achieved": float(np.mean(goals)),
        "mean_policy_calls": float(np.mean(calls)),
        "mean_selected_action_steps": float(np.mean(chosen_chunks)),
        "choice_counts": choice_counts,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
