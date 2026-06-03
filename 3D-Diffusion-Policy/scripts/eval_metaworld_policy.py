import argparse
import json
import os
import pathlib
import sys

import dill
import numpy as np
import torch

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
os.chdir(str(ROOT_DIR))

from train import TrainDP3Workspace  # noqa: E402
from diffusion_policy_3d.common.pytorch_util import dict_apply  # noqa: E402
from diffusion_policy_3d.env import MetaWorldEnv  # noqa: E402
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper  # noqa: E402
from diffusion_policy_3d.policy.adaptive_action import get_adaptive_metrics  # noqa: E402


def load_policy(checkpoint, device, num_inference_steps=None, n_overlap=0):
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    workspace = TrainDP3Workspace(payload["cfg"])
    workspace.load_payload(payload)
    policy = workspace.ema_model if workspace.cfg.training.use_ema else workspace.model
    if num_inference_steps is not None:
        policy.num_inference_steps = int(num_inference_steps)
    if n_overlap > 0:
        policy.n_overlap = int(n_overlap)
    policy.eval()
    policy.to(device)
    return policy


def make_env(task, n_obs_steps, n_action_steps, max_steps, device, use_point_crop=True):
    env_device = str(device)
    if env_device == "cuda":
        env_device = "cuda:0"
    return MultiStepWrapper(
        MetaWorldEnv(
            task_name=task,
            device=env_device,
            use_point_crop=use_point_crop,
            num_points=512,
        ),
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps,
        reward_agg_method="sum",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--n_overlap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy = load_policy(
        args.checkpoint,
        device,
        num_inference_steps=args.num_inference_steps,
        n_overlap=args.n_overlap,
    )
    env = make_env(
        args.task,
        int(policy.n_obs_steps),
        int(policy.n_action_steps),
        args.max_steps,
        device,
    )

    successes = []
    rewards = []
    calls = []
    mean_steps = []

    for episode in range(args.episodes):
        obs = env.reset()
        policy.reset()
        committed = None
        done = False
        reward_sum = 0.0
        success = False
        while not done:
            obs_t = dict_apply(
                dict(obs),
                lambda x: torch.from_numpy(x).to(device=device).unsqueeze(0),
            )
            with torch.no_grad():
                leftover = None
                if committed is not None:
                    leftover = torch.from_numpy(committed).to(
                        device=device, dtype=policy.dtype).unsqueeze(0)
                action_dict = policy.predict_action({
                    "point_cloud": obs_t["point_cloud"],
                    "agent_pos": obs_t["agent_pos"],
                }, leftover_actions=leftover)
            if args.n_overlap > 0:
                start = int(action_dict["action_start"][0].detach().cpu())
                output = action_dict["action_pred"][:, start:].detach().cpu().numpy().squeeze(0)
                n_fresh = int(action_dict.get(
                    "n_fresh",
                    torch.tensor([max(policy.n_action_steps - args.n_overlap, 1)])
                )[0].detach().cpu())
                if committed is None:
                    n_fresh = min(n_fresh, max(len(output) - args.n_overlap, 1))
                    action = output[:n_fresh]
                    committed = output[n_fresh:n_fresh + args.n_overlap]
                else:
                    fresh = output[len(committed):]
                    n_fresh = min(n_fresh, len(fresh))
                    action = np.concatenate([committed, fresh[:n_fresh]], axis=0)
                    committed = fresh[n_fresh:n_fresh + args.n_overlap]
                    if len(committed) == 0:
                        committed = None
            else:
                action = action_dict["action"].detach().cpu().numpy().squeeze(0)

            obs, reward, done, info = env.step(action)
            reward_sum += float(np.asarray(reward).sum())
            success = success or bool(np.asarray(info["success"]).max())
            done = bool(np.all(done))

        metrics = get_adaptive_metrics(policy)
        successes.append(float(success))
        rewards.append(reward_sum)
        if metrics:
            calls.append(metrics["policy_calls"])
            mean_steps.append(metrics["mean_selected_action_steps"])
        print(
            f"episode={episode:03d} success={int(success)} "
            f"reward={reward_sum:.1f} calls={calls[-1] if calls else 0}",
            flush=True,
        )

    result = {
        "task": args.task,
        "episodes": args.episodes,
        "mean_success_rates": float(np.mean(successes)),
        "mean_traj_rewards": float(np.mean(rewards)),
        "mean_policy_calls": float(np.mean(calls)) if calls else 0.0,
        "mean_selected_action_steps": float(np.mean(mean_steps)) if mean_steps else 0.0,
        "n_overlap": int(args.n_overlap),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
