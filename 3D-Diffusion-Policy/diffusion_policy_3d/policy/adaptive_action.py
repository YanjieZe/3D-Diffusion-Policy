import torch


def init_adaptive_action_policy(policy, mode="fixed", min_action_steps=1,
                                mid_action_steps=None, max_action_steps=None,
                                low_threshold=0.03, high_threshold=0.08,
                                overlap_alpha=0.5, schedule_boundaries=None,
                                schedule_action_steps=None,
                                uncertainty_samples=1,
                                phase_selector_path=None,
                                phase_selector_steps=None):
    policy.execution_mode = mode
    policy.min_action_steps = int(min_action_steps)
    policy.mid_action_steps = mid_action_steps
    policy.max_action_steps = max_action_steps
    policy.low_threshold = float(low_threshold)
    policy.high_threshold = float(high_threshold)
    policy.overlap_alpha = float(overlap_alpha)
    policy.schedule_boundaries = schedule_boundaries or [40, 70]
    policy.schedule_action_steps = schedule_action_steps or None
    policy.uncertainty_samples = max(1, int(uncertainty_samples))
    policy.phase_selector_path = phase_selector_path
    policy.phase_selector_steps = phase_selector_steps or schedule_action_steps or None
    policy.__dict__["_phase_selector"] = None
    policy.__dict__["_phase_selector_mean"] = None
    policy.__dict__["_phase_selector_std"] = None
    if phase_selector_path:
        payload = torch.load(phase_selector_path, map_location="cpu")
        input_dim = int(payload["input_dim"])
        hidden_dim = int(payload.get("hidden_dim", 64))
        output_dim = int(payload["output_dim"])
        selector = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        selector.load_state_dict(payload["state_dict"])
        selector.eval()
        policy.__dict__["_phase_selector"] = selector
        policy.__dict__["_phase_selector_mean"] = payload["state_mean"].float()
        policy.__dict__["_phase_selector_std"] = payload["state_std"].float()
        policy.phase_selector_steps = payload.get(
            "chunk_steps", policy.phase_selector_steps)
    reset_adaptive_action_policy(policy)


def reset_adaptive_action_policy(policy):
    policy._prev_action_pred = None
    policy._last_action_steps = None
    policy._num_policy_calls = 0
    policy._executed_action_steps = 0
    policy._selected_action_steps = []
    policy._adaptive_scores = []
    policy._overlap_scores = []
    policy._uncertainty_scores = []
    policy._phase_classes = []
    policy._branch_selector_probs = []


def _plan_curvature_score(action_plan):
    if action_plan.shape[1] <= 1:
        return torch.zeros(action_plan.shape[0], device=action_plan.device)

    velocity = action_plan[:, 1:] - action_plan[:, :-1]
    velocity_score = torch.linalg.norm(velocity, dim=-1).mean(dim=-1)

    if action_plan.shape[1] <= 2:
        return velocity_score

    curvature = velocity[:, 1:] - velocity[:, :-1]
    curvature_score = torch.linalg.norm(curvature, dim=-1).mean(dim=-1)
    return velocity_score + 0.5 * curvature_score


def _overlap_plan(policy, action_pred, start, max_steps):
    prev_action_pred = policy._prev_action_pred
    last_action_steps = policy._last_action_steps
    if prev_action_pred is None or last_action_steps is None:
        return None

    prev_start = start + last_action_steps
    prev_end = min(prev_start + max_steps, prev_action_pred.shape[1])
    if prev_start >= prev_end:
        return None

    return prev_action_pred[:, prev_start:prev_end].to(
        device=action_pred.device, dtype=action_pred.dtype)


def build_branch_selector_features(policy, action_pred, start, max_steps,
                                   phase_obs=None):
    """Features shared by simulator-label training and policy inference."""
    batch = action_pred.shape[0]
    device = action_pred.device
    dtype = action_pred.dtype
    plan = action_pred[:, start:start + max_steps]
    if plan.shape[1] == 0:
        plan = action_pred[:, -1:].clone()

    if phase_obs is None:
        phase_obs = torch.zeros((batch, 0), device=device, dtype=dtype)
    else:
        phase_obs = phase_obs.to(device=device, dtype=dtype)

    executed = torch.full(
        (batch, 1),
        float(getattr(policy, "_executed_action_steps", 0)) / 100.0,
        device=device,
        dtype=dtype)

    first_action = plan[:, 0]
    last_action = plan[:, -1]
    mean_action = plan.mean(dim=1)
    std_action = plan.std(dim=1, unbiased=False)
    mean_abs = plan.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
    max_abs = plan.abs().amax(dim=(1, 2), keepdim=False).unsqueeze(-1)

    if plan.shape[1] > 1:
        velocity = plan[:, 1:] - plan[:, :-1]
        velocity_norm = torch.linalg.norm(velocity, dim=-1).mean(
            dim=-1, keepdim=True)
    else:
        velocity = None
        velocity_norm = torch.zeros((batch, 1), device=device, dtype=dtype)

    if velocity is not None and velocity.shape[1] > 1:
        curvature = velocity[:, 1:] - velocity[:, :-1]
        curvature_norm = torch.linalg.norm(curvature, dim=-1).mean(
            dim=-1, keepdim=True)
    else:
        curvature_norm = torch.zeros((batch, 1), device=device, dtype=dtype)

    return torch.cat([
        phase_obs,
        executed,
        first_action,
        last_action,
        mean_action,
        std_action,
        mean_abs,
        max_abs,
        velocity_norm,
        curvature_norm,
    ], dim=-1)


def select_adaptive_action(policy, action_pred, start, uncertainty_profile=None,
                           phase_obs=None):
    mode = getattr(policy, "execution_mode", "fixed")
    default_steps = int(policy.n_action_steps)
    max_cfg = getattr(policy, "max_action_steps", None)
    max_steps = int(max_cfg) if max_cfg is not None else default_steps
    if mode == "fixed":
        max_steps = min(max_steps, default_steps)
    max_steps = max(1, min(max_steps, action_pred.shape[1] - start))

    min_steps = max(1, min(int(getattr(policy, "min_action_steps", 1)), max_steps))
    mid_cfg = getattr(policy, "mid_action_steps", None)
    mid_steps = int(mid_cfg) if mid_cfg is not None else max(min_steps, max_steps // 2)
    mid_steps = max(min_steps, min(mid_steps, max_steps))

    candidate = action_pred[:, start:start + max_steps].clone()
    prev_overlap = _overlap_plan(policy, action_pred, start, max_steps)

    overlap_score = torch.zeros(action_pred.shape[0], device=action_pred.device)
    if prev_overlap is not None:
        overlap_len = min(prev_overlap.shape[1], candidate.shape[1])
        if overlap_len > 0:
            current_overlap = candidate[:, :overlap_len]
            overlap_delta = current_overlap - prev_overlap[:, :overlap_len]
            overlap_score = torch.linalg.norm(overlap_delta, dim=-1).mean(dim=-1)
            if mode in ("overlap", "overlap_dynamic", "overlap_schedule"):
                alpha = float(getattr(policy, "overlap_alpha", 0.5))
                candidate[:, :overlap_len] = (
                    alpha * current_overlap + (1.0 - alpha) * prev_overlap[:, :overlap_len])

    curvature_score = _plan_curvature_score(candidate)
    score = curvature_score
    if uncertainty_profile is not None:
        uncertainty_profile = uncertainty_profile[:, start:start + max_steps].to(
            device=action_pred.device, dtype=action_pred.dtype)
        uncertainty_prefix = []
        for steps in (min_steps, mid_steps, max_steps):
            uncertainty_prefix.append(
                float(uncertainty_profile[:, :steps].mean().detach().cpu()))
        policy._uncertainty_scores.append(uncertainty_prefix[-1])
    else:
        uncertainty_prefix = None

    if mode in ("dynamic", "overlap_dynamic"):
        score = score + overlap_score

    if mode in ("uncertainty", "overlap_uncertainty", "best_of_n_uncertainty"):
        if uncertainty_prefix is None:
            mean_score = float(score.mean().detach().cpu())
            selected_steps = max_steps
        else:
            low = float(getattr(policy, "low_threshold", 0.03))
            high = float(getattr(policy, "high_threshold", 0.08))
            min_u, mid_u, max_u = uncertainty_prefix
            # Choose the longest prefix whose diffusion samples still agree.
            if max_u <= low:
                selected_steps = max_steps
            elif mid_u <= high:
                selected_steps = mid_steps
            else:
                selected_steps = min_steps
            mean_score = max_u
    elif mode in ("phase_mlp", "branch_selector"):
        selector = getattr(policy, "_phase_selector", None)
        if selector is None:
            selected_steps = max_steps
        else:
            feature_device = action_pred.device
            selector = selector.to(device=feature_device)
            mean = policy._phase_selector_mean.to(device=feature_device)
            std = policy._phase_selector_std.to(device=feature_device)
            if mode == "branch_selector":
                selector_input = build_branch_selector_features(
                    policy, action_pred, start, max_steps, phase_obs=phase_obs)
            else:
                selector_input = phase_obs
                if phase_obs is None:
                    selector_input = torch.zeros(
                        (action_pred.shape[0], 0),
                        device=feature_device,
                        dtype=action_pred.dtype)
                if mean.shape[-1] == 1:
                    selector_input = torch.full(
                        (action_pred.shape[0], 1),
                        float(getattr(policy, "_executed_action_steps", 0)) / 100.0,
                        device=feature_device,
                        dtype=action_pred.dtype)
                elif mean.shape[-1] == selector_input.shape[-1] + 1:
                    executed = torch.full(
                        (action_pred.shape[0], 1),
                        float(getattr(policy, "_executed_action_steps", 0)) / 100.0,
                        device=feature_device,
                        dtype=action_pred.dtype)
                    selector_input = torch.cat([selector_input, executed], dim=-1)
            selector_input = (selector_input - mean) / std.clamp_min(1e-6)
            with torch.no_grad():
                logits = selector(selector_input)
                phase_class = torch.argmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)
            class_idx = int(phase_class[0].detach().cpu())
            phase_steps = getattr(policy, "phase_selector_steps", None)
            if phase_steps is None:
                phase_steps = [max_steps, mid_steps, min_steps]
            selected_steps = int(phase_steps[min(class_idx, len(phase_steps) - 1)])
            selected_steps = max(1, min(selected_steps, max_steps))
            policy._phase_classes.append(class_idx)
            policy._branch_selector_probs.append(
                float(probs.max(dim=-1).values.mean().detach().cpu()))
        mean_score = float(score.mean().detach().cpu())
    elif mode in ("dynamic", "overlap_dynamic"):
        mean_score = float(score.mean().detach().cpu())
        if mean_score >= float(getattr(policy, "high_threshold", 0.08)):
            selected_steps = min_steps
        elif mean_score >= float(getattr(policy, "low_threshold", 0.03)):
            selected_steps = mid_steps
        else:
            selected_steps = max_steps
    elif mode in ("schedule", "overlap_schedule"):
        schedule_steps = getattr(policy, "schedule_action_steps", None)
        if schedule_steps is None:
            schedule_steps = [max_steps, mid_steps, min_steps]
        schedule_steps = [max(1, min(int(x), max_steps)) for x in schedule_steps]
        boundaries = [int(x) for x in getattr(policy, "schedule_boundaries", [40, 70])]
        executed_steps = int(getattr(policy, "_executed_action_steps", 0))
        if executed_steps < boundaries[0]:
            selected_steps = schedule_steps[0]
        elif executed_steps < boundaries[1]:
            selected_steps = schedule_steps[min(1, len(schedule_steps) - 1)]
        else:
            selected_steps = schedule_steps[-1]
        mean_score = float(score.mean().detach().cpu())
    else:
        selected_steps = max_steps
        mean_score = float(score.mean().detach().cpu())

    action = candidate[:, :selected_steps]

    policy._num_policy_calls += 1
    policy._last_action_steps = selected_steps
    policy._executed_action_steps += selected_steps
    policy._prev_action_pred = action_pred.detach()
    policy._selected_action_steps.append(selected_steps)
    policy._adaptive_scores.append(mean_score)
    policy._overlap_scores.append(float(overlap_score.mean().detach().cpu()))

    return {
        "action": action,
        "action_steps": torch.full(
            (action_pred.shape[0],), selected_steps,
            device=action_pred.device, dtype=torch.long),
        "adaptive_score": torch.full(
            (action_pred.shape[0],), mean_score,
            device=action_pred.device, dtype=action_pred.dtype),
    }


def get_adaptive_metrics(policy):
    selected = getattr(policy, "_selected_action_steps", [])
    scores = getattr(policy, "_adaptive_scores", [])
    overlap_scores = getattr(policy, "_overlap_scores", [])
    uncertainty_scores = getattr(policy, "_uncertainty_scores", [])

    if len(selected) == 0:
        return {}

    return {
        "policy_calls": int(getattr(policy, "_num_policy_calls", 0)),
        "mean_selected_action_steps": float(sum(selected) / len(selected)),
        "min_selected_action_steps": int(min(selected)),
        "max_selected_action_steps": int(max(selected)),
        "mean_adaptive_score": float(sum(scores) / len(scores)) if scores else 0.0,
        "mean_overlap_score": (
            float(sum(overlap_scores) / len(overlap_scores)) if overlap_scores else 0.0),
        "mean_uncertainty_score": (
            float(sum(uncertainty_scores) / len(uncertainty_scores))
            if uncertainty_scores else 0.0),
        "mean_phase_class": (
            float(sum(getattr(policy, "_phase_classes", [])) /
                  len(getattr(policy, "_phase_classes", [])))
            if getattr(policy, "_phase_classes", []) else 0.0),
        "mean_branch_selector_confidence": (
            float(sum(getattr(policy, "_branch_selector_probs", [])) /
                  len(getattr(policy, "_branch_selector_probs", [])))
            if getattr(policy, "_branch_selector_probs", []) else 0.0),
    }
