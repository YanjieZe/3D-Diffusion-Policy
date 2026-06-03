import argparse
import json
import os
import pathlib
import sys

import dill
import torch
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
os.chdir(str(ROOT_DIR))

from train import TrainDP3Workspace  # noqa: E402
from diffusion_policy_3d.common.pytorch_util import dict_apply  # noqa: E402
from diffusion_policy_3d.dataset.base_dataset import BaseDataset  # noqa: E402
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--temporal_loss_weight", type=float, default=2.0)
    parser.add_argument("--temporal_loss_center", type=float, default=0.55)
    parser.add_argument("--temporal_loss_width", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_train_steps", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    payload = torch.load(open(args.checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg.policy.temporal_loss_weight = args.temporal_loss_weight
    cfg.policy.temporal_loss_center = args.temporal_loss_center
    cfg.policy.temporal_loss_width = args.temporal_loss_width

    workspace = TrainDP3Workspace(cfg, output_dir=args.output_dir)
    workspace.load_payload(payload)
    policy = workspace.model
    policy.temporal_loss_weight = args.temporal_loss_weight
    policy.temporal_loss_center = args.temporal_loss_center
    policy.temporal_loss_width = args.temporal_loss_width

    dataset: BaseDataset = __import__("hydra").utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    if workspace.ema_model is not None:
        workspace.ema_model.set_normalizer(normalizer)
        workspace.ema_model.temporal_loss_weight = args.temporal_loss_weight
        workspace.ema_model.temporal_loss_center = args.temporal_loss_center
        workspace.ema_model.temporal_loss_width = args.temporal_loss_width

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        persistent_workers=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy.to(device)
    if workspace.ema_model is not None:
        workspace.ema_model.to(device)
        ema = EMAModel(model=workspace.ema_model, update_after_step=0)
    else:
        ema = None
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-6)

    history = []
    global_step = 0
    for epoch in range(args.epochs):
        policy.train()
        losses = []
        with tqdm.tqdm(dataloader, desc=f"temporal-ft epoch {epoch}", leave=False) as tepoch:
            for step, batch in enumerate(tepoch):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                loss, loss_dict = policy.compute_loss(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if ema is not None:
                    ema.step(policy)
                losses.append(float(loss.detach().cpu()))
                global_step += 1
                tepoch.set_postfix(loss=losses[-1], refresh=False)
                if args.max_train_steps is not None and step + 1 >= args.max_train_steps:
                    break
        epoch_loss = sum(losses) / max(1, len(losses))
        history.append({"epoch": epoch, "train_loss": epoch_loss})
        print(f"epoch={epoch} train_loss={epoch_loss:.6f}", flush=True)

    workspace.global_step = global_step
    workspace.epoch = int(payload.get("pickles", {}).get("epoch", b"0") != b"0") + args.epochs
    latest_path = os.path.join(args.output_dir, "checkpoints", "latest.ckpt")
    workspace.save_checkpoint(path=latest_path)
    with open(os.path.join(args.output_dir, "temporal_finetune_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved {latest_path}")


if __name__ == "__main__":
    main()
