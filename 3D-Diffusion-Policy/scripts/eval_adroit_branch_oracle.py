import argparse
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
from scripts.collect_adroit_branch_data import (  # noqa: E402
    goal_value,
    near_best_label,
    score_branch,
    stack_obs,
)


def load_policy(checkpoint, device):
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    workspace = TrainDP3Workspace(payload["cfg"])
    workspace.load_payload(payload)
    policy = workspace.ema_model if workspace.cfg.training.use_ema else workspace.model
    policy.execution_mode = "fixed"
    policy.eval()
    policy.to(device)
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=["door", "hammer"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--chunks", type=int, nargs="+", default=[2, 4, 8, 12])
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument(
        "--tie_break", choices=["argmax", "shortest", "longest"],
        default="longest")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_point_crop", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.checkpoint, device)
    env = MujocoPointcloudWrapperAdroit(
        env=AdroitEnv(env_name=args.task, use_point_cloud=True),
        env_name=f"adroit_{args.task}",
        use_point_crop=not args.no_point_crop,
    )
    n_obs_steps = int(policy.n_obs_steps)

    successes = []
    goals = []
    calls = []
    selected_chunks = []

    for episode in range(args.episodes):
        obs = env.reset()
        history = deque([obs], maxlen=n_obs_steps)
        policy.reset()
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
            with torch.no_grad():
                action_dict = policy.predict_action({
                    "point_cloud": obs_t["point_cloud"],
                    "agent_pos": obs_t["agent_pos"],
                })
            action_pred = action_dict["action_pred"].detach().cpu().numpy().squeeze(0)
            start = n_obs_steps - 1
            plan = action_pred[start:start + max(args.chunks)]
            state = env.get_env_state()
            branch_scores = [
                score_branch(env, state, plan, min(chunk, len(plan)))
                for chunk in args.chunks
            ]
            label = near_best_label(
                branch_scores, args.chunks, args.margin, args.tie_break)
            chunk = min(int(args.chunks[label]), len(plan))
            env.set_env_state(state)

            episode_calls += 1
            selected_chunks.append(chunk)
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

    result = {
        "task": args.task,
        "execution_mode": "branch_oracle",
        "episodes": args.episodes,
        "chunks": args.chunks,
        "mean_success_rates": float(np.mean(successes)),
        "mean_n_goal_achieved": float(np.mean(goals)),
        "mean_policy_calls": float(np.mean(calls)),
        "mean_selected_action_steps": float(np.mean(selected_chunks)),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
