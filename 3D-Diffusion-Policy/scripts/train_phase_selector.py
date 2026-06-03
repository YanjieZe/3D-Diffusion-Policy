import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import zarr


def build_phase_dataset(zarr_path, boundaries, include_time=False):
    root = zarr.open(zarr_path, mode="r")
    states = root["data/state"][:].astype(np.float32)
    episode_ends = root["meta/episode_ends"][:]
    labels = np.zeros((states.shape[0],), dtype=np.int64)
    progress = np.zeros((states.shape[0], 1), dtype=np.float32)
    start = 0
    for end in episode_ends:
        t = np.arange(end - start)
        labels[start:end] = np.where(
            t < boundaries[0], 0, np.where(t < boundaries[1], 1, 2))
        progress[start:end, 0] = t.astype(np.float32) / 100.0
        start = end
    if include_time:
        states = np.concatenate([states, progress], axis=-1)
    return states, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--boundaries", type=int, nargs=2, default=[40, 70])
    parser.add_argument("--chunk_steps", type=int, nargs="+", default=[8, 4, 2])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include_time", action="store_true")
    parser.add_argument("--time_only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    states, labels = build_phase_dataset(
        args.zarr_path, args.boundaries, include_time=args.include_time)
    if args.time_only:
        states = states[:, -1:]
    mean = torch.from_numpy(states.mean(axis=0)).float()
    std = torch.from_numpy(states.std(axis=0)).float().clamp_min(1e-6)
    x = (torch.from_numpy(states).float() - mean) / std
    y = torch.from_numpy(labels).long()

    model = nn.Sequential(
        nn.Linear(x.shape[-1], args.hidden_dim),
        nn.ReLU(),
        nn.Linear(args.hidden_dim, 3),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(args.epochs):
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        acc = (model(x).argmax(dim=-1) == y).float().mean().item()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": x.shape[-1],
        "hidden_dim": args.hidden_dim,
        "output_dim": 3,
        "state_mean": mean,
        "state_std": std,
        "chunk_steps": list(args.chunk_steps),
        "boundaries": list(args.boundaries),
        "train_accuracy": acc,
        "include_time": bool(args.include_time),
        "time_only": bool(args.time_only),
    }, args.output)
    print(f"saved {args.output}")
    print(f"train_accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
