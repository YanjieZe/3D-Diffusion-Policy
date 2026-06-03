import argparse
import copy
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
from diffusion_policy_3d.policy.adaptive_action import (  # noqa: E402
    build_branch_selector_features,
)


def load_policy(checkpoint, device):
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    workspace = TrainDP3Workspace(payload["cfg"])
    workspace.load_payload(payload)
    policy = workspace.ema_model if workspace.cfg.training.use_ema else workspace.model
    policy.eval()
    policy.to(device)
    policy.execution_mode = "fixed"
    return policy


def make_env(task_name, use_point_crop=True):
    return MujocoPointcloudWrapperAdroit(
        env=AdroitEnv(env_name=task_name, use_point_cloud=True),
        env_name=f"adroit_{task_name}",
        use_point_crop=use_point_crop,
    )


def stack_obs(history, n_obs_steps):
    result = {}
    for key in history[-1].keys():
        values = [obs[key] for obs in history]
        if len(values) < n_obs_steps:
            values = [values[0]] * (n_obs_steps - len(values)) + values
        result[key] = np.stack(values[-n_obs_steps:], axis=0)
    return result


def goal_value(info):
    if "goal_achieved" in info:
        return float(np.asarray(info["goal_achieved"]).max())
    if "n_goal_achieved" in info:
        return float(info["n_goal_achieved"])
    return 0.0


def score_branch(env, state, actions, chunk):
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
    mean_reward = reward_sum / max(1.0, float(chunk))
    mean_goal = goal_sum / max(1.0, float(chunk))
    progress_score = 20.0 * final_goal + 5.0 * mean_goal + 5.0 * mean_reward
    efficiency_penalty = 0.30 * float(chunk)
    return success_bonus + progress_score - efficiency_penalty


def near_best_label(branch_scores, chunks, margin, tie_break):
    branch_scores = np.asarray(branch_scores, dtype=np.float32)
    best_score = float(branch_scores.max())
    eligible = np.flatnonzero(branch_scores >= best_score - float(margin))
    if eligible.size == 0:
        return int(branch_scores.argmax())
    eligible_chunks = np.asarray(chunks, dtype=np.int64)[eligible]
    if tie_break == "shortest":
        return int(eligible[np.argmin(eligible_chunks)])
    if tie_break == "longest":
        return int(eligible[np.argmax(eligible_chunks)])
    return int(branch_scores.argmax())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=["door", "hammer"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--chunks", type=int, nargs="+", default=[2, 4, 8, 12])
    parser.add_argument("--shortest_margin", type=float, default=2.0)
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
    env = make_env(args.task, use_point_crop=not args.no_point_crop)
    n_obs_steps = int(policy.n_obs_steps)

    features = []
    labels = []
    scores = []
    episode_ids = []
    env_steps = []

    for episode in range(args.episodes):
        obs = env.reset()
        history = deque([obs], maxlen=n_obs_steps)
        policy.reset()
        done = False
        step_count = 0

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

            executed_before = int(getattr(policy, "_executed_action_steps", 0))
            with torch.no_grad():
                action_dict = policy.predict_action(obs_input)

            np_action = action_dict["action"].detach().cpu().numpy().squeeze(0)
            action_pred = action_dict["action_pred"]
            nobs = policy.normalizer.normalize(obs_input)
            phase_obs = nobs["agent_pos"][:, n_obs_steps - 1]

            old_executed = int(getattr(policy, "_executed_action_steps", 0))
            policy._executed_action_steps = executed_before
            with torch.no_grad():
                feat = build_branch_selector_features(
                    policy,
                    action_pred,
                    n_obs_steps - 1,
                    max(args.chunks),
                    phase_obs=phase_obs,
                )
            policy._executed_action_steps = old_executed

            state = env.get_env_state()
            plan = action_pred.detach().cpu().numpy().squeeze(0)[
                n_obs_steps - 1:n_obs_steps - 1 + max(args.chunks)
            ]
            branch_scores = [
                score_branch(env, state, plan, min(chunk, len(plan)))
                for chunk in args.chunks
            ]
            env.set_env_state(copy.deepcopy(state))

            features.append(feat.detach().cpu().numpy().squeeze(0))
            labels.append(near_best_label(
                branch_scores, args.chunks, args.shortest_margin, args.tie_break))
            scores.append(branch_scores)
            episode_ids.append(episode)
            env_steps.append(step_count)

            for act in np_action:
                obs, _, done, _ = env.step(act)
                history.append(obs)
                step_count += 1
                if done or step_count >= args.max_steps:
                    break

        print(
            f"episode={episode:03d} states={len(labels)} "
            f"steps={step_count} done={done}",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        scores=np.asarray(scores, dtype=np.float32),
        chunks=np.asarray(args.chunks, dtype=np.int64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        env_steps=np.asarray(env_steps, dtype=np.int64),
        checkpoint=str(args.checkpoint),
        task=str(args.task),
    )
    counts = np.bincount(np.asarray(labels), minlength=len(args.chunks))
    print(f"saved {args.output}")
    print(f"label_counts={dict(zip(args.chunks, counts.tolist()))}")


if __name__ == "__main__":
    main()
