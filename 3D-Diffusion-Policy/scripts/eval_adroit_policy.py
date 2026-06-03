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
from diffusion_policy_3d.env import AdroitEnv  # noqa: E402
from diffusion_policy_3d.gym_util.mjpc_diffusion_wrapper import (  # noqa: E402
    MujocoPointcloudWrapperAdroit,
)
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper  # noqa: E402
from diffusion_policy_3d.policy.adaptive_action import (  # noqa: E402
    get_adaptive_metrics,
    init_adaptive_action_policy,
)


def load_policy(checkpoint, device, args):
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    workspace = TrainDP3Workspace(payload["cfg"])
    workspace.load_payload(payload)
    policy = workspace.ema_model if workspace.cfg.training.use_ema else workspace.model
    if args.num_inference_steps is not None:
        policy.num_inference_steps = int(args.num_inference_steps)
    init_adaptive_action_policy(
        policy,
        mode=args.execution_mode,
        min_action_steps=args.min_action_steps,
        mid_action_steps=args.mid_action_steps,
        max_action_steps=args.max_action_steps,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        schedule_boundaries=args.schedule_boundaries,
        schedule_action_steps=args.schedule_action_steps,
        uncertainty_samples=args.uncertainty_samples,
        phase_selector_path=args.selector,
        phase_selector_steps=args.selector_steps,
    )
    policy.eval()
    policy.to(device)
    return policy


def make_env(task, n_obs_steps, n_action_steps, max_steps, use_point_crop=True):
    return MultiStepWrapper(
        MujocoPointcloudWrapperAdroit(
            env=AdroitEnv(env_name=task, use_point_cloud=True),
            env_name=f"adroit_{task}",
            use_point_crop=use_point_crop,
        ),
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps,
        reward_agg_method="sum",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=["door", "hammer", "pen"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execution_mode", default="fixed")
    parser.add_argument("--selector", default=None)
    parser.add_argument("--selector_steps", type=int, nargs="+", default=None)
    parser.add_argument("--min_action_steps", type=int, default=1)
    parser.add_argument("--mid_action_steps", type=int, default=4)
    parser.add_argument("--max_action_steps", type=int, default=None)
    parser.add_argument("--low_threshold", type=float, default=0.03)
    parser.add_argument("--high_threshold", type=float, default=0.08)
    parser.add_argument("--schedule_boundaries", type=int, nargs="+", default=[40, 70])
    parser.add_argument("--schedule_action_steps", type=int, nargs="+", default=None)
    parser.add_argument("--uncertainty_samples", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--n_overlap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_point_crop", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.checkpoint, device, args)
    env = make_env(
        args.task,
        int(policy.n_obs_steps),
        int(policy.n_action_steps),
        args.max_steps,
        use_point_crop=not args.no_point_crop,
    )

    all_success = []
    all_goal = []
    all_calls = []
    all_mean_steps = []
    all_confidence = []

    for episode in range(args.episodes):
        obs = env.reset()
        policy.reset()
        if args.n_overlap > 0:
            policy.n_overlap = int(args.n_overlap)
        committed = None
        done = False
        goal_count = 0.0
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
                full = action_dict["action_pred"][:, start:]
                output = full.detach().cpu().numpy().squeeze(0)
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
                    action = np.concatenate([committed, fresh[:n_fresh]], axis=0)
                    committed = fresh[n_fresh:n_fresh + args.n_overlap]
                    if len(committed) == 0:
                        committed = None
            else:
                action = action_dict["action"].detach().cpu().numpy().squeeze(0)
            obs, _, done, info = env.step(action)
            if "goal_achieved" in info:
                goal_count += float(np.asarray(info["goal_achieved"]).sum())
            done = bool(np.all(done))

        success = float(np.asarray(info.get("goal_achieved", 0)).max())
        metrics = get_adaptive_metrics(policy)
        all_success.append(success)
        all_goal.append(goal_count)
        if metrics:
            all_calls.append(metrics["policy_calls"])
            all_mean_steps.append(metrics["mean_selected_action_steps"])
            all_confidence.append(metrics.get("mean_branch_selector_confidence", 0.0))
        print(
            f"episode={episode:03d} success={success:.0f} "
            f"goal={goal_count:.1f} calls={all_calls[-1] if all_calls else 0}",
            flush=True,
        )

    result = {
        "task": args.task,
        "execution_mode": args.execution_mode,
        "episodes": args.episodes,
        "mean_success_rates": float(np.mean(all_success)),
        "mean_n_goal_achieved": float(np.mean(all_goal)),
        "n_overlap": int(args.n_overlap),
        "mean_policy_calls": float(np.mean(all_calls)) if all_calls else 0.0,
        "mean_selected_action_steps": (
            float(np.mean(all_mean_steps)) if all_mean_steps else 0.0),
        "mean_branch_selector_confidence": (
            float(np.mean(all_confidence)) if all_confidence else 0.0),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
