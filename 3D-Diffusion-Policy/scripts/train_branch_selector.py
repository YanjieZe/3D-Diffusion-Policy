import argparse
import os

import numpy as np
import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.dataset, allow_pickle=True)
    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    chunks = data["chunks"].astype(np.int64).tolist()

    order = np.random.permutation(features.shape[0])
    val_size = max(1, int(features.shape[0] * args.val_ratio))
    val_idx = order[:val_size]
    train_idx = order[val_size:]
    if train_idx.size == 0:
        raise ValueError("need more branch samples; train split is empty")

    mean = torch.from_numpy(features[train_idx].mean(axis=0)).float()
    std = torch.from_numpy(features[train_idx].std(axis=0)).float().clamp_min(1e-6)
    x = (torch.from_numpy(features).float() - mean) / std
    y = torch.from_numpy(labels).long()

    train_x = x[train_idx]
    train_y = y[train_idx]
    val_x = x[val_idx]
    val_y = y[val_idx]

    class_counts = torch.bincount(train_y, minlength=len(chunks)).float()
    class_weights = class_counts.sum() / class_counts.clamp_min(1.0)
    class_weights = class_weights / class_weights.mean()

    model = nn.Sequential(
        nn.Linear(x.shape[-1], args.hidden_dim),
        nn.ReLU(),
        nn.Linear(args.hidden_dim, len(chunks)),
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    for _ in range(args.epochs):
        pred = model(train_x)
        loss = loss_fn(pred, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_acc = (model(train_x).argmax(dim=-1) == train_y).float().mean().item()
        val_acc = (model(val_x).argmax(dim=-1) == val_y).float().mean().item()
        pred_counts = torch.bincount(
            model(x).argmax(dim=-1), minlength=len(chunks)).tolist()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": x.shape[-1],
        "hidden_dim": args.hidden_dim,
        "output_dim": len(chunks),
        "state_mean": mean,
        "state_std": std,
        "chunk_steps": chunks,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "class_counts": class_counts.tolist(),
        "pred_counts": pred_counts,
        "source_dataset": args.dataset,
        "selector_type": "simulator_branch",
    }, args.output)
    print(f"saved {args.output}")
    print(f"train_accuracy={train_acc:.4f}")
    print(f"val_accuracy={val_acc:.4f}")
    print(f"class_counts={dict(zip(chunks, class_counts.int().tolist()))}")
    print(f"pred_counts={dict(zip(chunks, pred_counts))}")


if __name__ == "__main__":
    main()
