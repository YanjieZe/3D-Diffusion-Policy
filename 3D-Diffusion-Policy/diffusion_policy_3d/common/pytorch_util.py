from typing import Dict, Callable, List
import collections
import torch
import torch.nn as nn


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return a per-sample mean over the entries selected by ``mask``.

    Diffusion inpainting losses mask out observed dimensions.  Reducing the
    masked tensor with a regular mean still divides by the *unmasked* tensor
    size, making the effective objective depend on how many dimensions are
    conditioned.  This helper keeps each sample's loss scale independent of
    that mask and handles an all-conditioned sample without NaNs.
    """

    if value.shape != mask.shape:
        raise ValueError(
            "value and mask must have identical shapes: "
            f"got value={tuple(value.shape)}, mask={tuple(mask.shape)}"
        )
    value_flat = value.reshape(value.shape[0], -1)
    mask_flat = mask.to(dtype=value.dtype).reshape(value.shape[0], -1)
    denominator = mask_flat.sum(dim=1).clamp_min(1.0)
    return (value_flat * mask_flat).sum(dim=1) / denominator


def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

def pad_remaining_dims(x, target):
    assert x.shape == target.shape[:len(x.shape)]
    return x.reshape(x.shape + (1,)*(len(target.shape) - len(x.shape)))

def dict_apply_split(
        x: Dict[str, torch.Tensor], 
        split_func: Callable[[torch.Tensor], Dict[str, torch.Tensor]]
        ) -> Dict[str, torch.Tensor]:
    results = collections.defaultdict(dict)
    for key, value in x.items():
        result = split_func(value)
        for k, v in result.items():
            results[k][key] = v
    return results

def dict_apply_reduce(
        x: List[Dict[str, torch.Tensor]],
        reduce_func: Callable[[List[torch.Tensor]], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key in x[0].keys():
        result[key] = reduce_func([x_[key] for x_ in x])
    return result


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer
