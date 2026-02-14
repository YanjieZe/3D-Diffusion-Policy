if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
from hydra.core.hydra_config import HydraConfig
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

def discretize_gripper_action_torch(gripper_value):
    """
    Discretize continuous gripper action to {-1, 0, 1} using torch operations
    
    Args:
        gripper_value: Tensor of continuous values from policy
    
    Returns:
        Tensor of discrete commands: -1.0 (open), 0.0 (hold), or 1.0 (close)
    """
    gripper_discrete = torch.zeros_like(gripper_value)
    gripper_discrete[gripper_value < -0.33] = -1.0
    gripper_discrete[gripper_value > 0.33] = 1.0
    # Values between -0.33 and 0.33 remain 0.0
    return gripper_discrete



class TrainDP3Workspace:
    include_keys = ["global_step", "epoch"]
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DP3 = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DP3 = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except:  # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False

        RUN_VALIDATION = True

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(
            f"dataset must be BaseDataset, got {type(dataset)}"
        )

        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=self.output_dir
        )

        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)

        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(
                train_dataloader,
                desc=f"Training epoch {self.epoch}",
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec,
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # compute loss
                    t1_1 = time.time()
                    raw_loss, loss_dict = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)
                    t1_4 = time.time()
                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        "train_loss": raw_loss_cpu,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()

                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = batch_idx == (len(train_dataloader) - 1)
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) and batch_idx >= (
                        cfg.training.max_train_steps - 1
                    ):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log["train_loss"] = train_loss

            # ========= eval for this epoch ==========
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()
            

            # run rollout
            if (
                (self.epoch % cfg.training.rollout_every) == 0
                and RUN_ROLLOUT
                and env_runner is not None
            ):
                t3 = time.time()
                runner_log = env_runner.run(policy, dataset=dataset)
                t4 = time.time()
                # log all
                step_log.update(runner_log)

            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(
                        val_dataloader,
                        desc=f"Validation epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(
                                batch, lambda x: x.to(device, non_blocking=True)
                            )
                            loss, loss_dict = self.model.compute_loss(batch)
                            val_losses.append(loss)
                            if (
                                cfg.training.max_val_steps is not None
                            ) and batch_idx >= (cfg.training.max_val_steps - 1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log["val_loss"] = val_loss

            # run diffusion sampling on a training batch
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    batch = dict_apply(
                        train_sampling_batch, lambda x: x.to(device, non_blocking=True)
                    )
                    obs_dict = batch["obs"]
                    gt_action = batch["action"]

                    result = policy.predict_action(obs_dict)
                    pred_action = result["action_pred"]
                    
                    # ========= GRIPPER-AWARE METRICS (6D vs 7D) =========
                    
                    # Determine action dimension
                    action_dim = gt_action.shape[-1]
                    
                    # 1. Overall MSE (baseline)
                    mse_overall = torch.nn.functional.mse_loss(pred_action, gt_action)
                    
                    if action_dim == 6:
                        # ===== 6D ACTION SPACE (ARM ONLY, NO GRIPPER) =====
                        cprint(f"\nTrain Action Metrics (Epoch {self.epoch}) - 6D (Arm Only):", "cyan")
                        cprint(f"  Overall MSE:           {mse_overall.item():.6f}", "cyan")
                        
                        # Per-joint MSE
                        mse_per_joint = torch.mean((pred_action - gt_action) ** 2, dim=(0, 1))
                        for i in range(6):
                            cprint(f"  Joint {i} MSE:          {mse_per_joint[i].item():.6f}", "cyan")
                        
                        # Log to wandb
                        step_log["train_action_mse_overall"] = mse_overall.item()
                        step_log["train_action_mse_error"] = mse_overall.item()  # Legacy
                        for i in range(6):
                            step_log[f"train_joint_{i}_mse"] = mse_per_joint[i].item()
                        
                    elif action_dim == 7 or action_dim == 13:
                        # ===== 7D/13D ACTION SPACE (ARM + GRIPPER) =====
                        # Determine gripper index
                        if action_dim == 7:
                            gripper_idx = 6  # [arm_joints(6), gripper(1)]
                        elif action_dim == 13:
                            gripper_idx = 12  # [eef_delta(6), arm_joints(6), gripper(1)]
                        else:
                            cprint(f"WARNING: Unexpected action_dim={action_dim}", "yellow")
                            gripper_idx = action_dim - 1
                        
                        # 2. MSE without gripper dimension
                        pred_action_no_gripper = torch.cat([
                            pred_action[..., :gripper_idx],
                            pred_action[..., gripper_idx+1:]
                        ], dim=-1)
                        gt_action_no_gripper = torch.cat([
                            gt_action[..., :gripper_idx],
                            gt_action[..., gripper_idx+1:]
                        ], dim=-1)
                        mse_no_gripper = torch.nn.functional.mse_loss(
                            pred_action_no_gripper, gt_action_no_gripper
                        )
                        
                        # 3. MSE with binarized gripper
                        pred_action_binarized = pred_action.clone()
                        pred_gripper = pred_action[..., gripper_idx]
                        pred_gripper_discrete = discretize_gripper_action_torch(pred_gripper)
                        pred_action_binarized[..., gripper_idx] = pred_gripper_discrete
                        
                        mse_binarized = torch.nn.functional.mse_loss(
                            pred_action_binarized, gt_action
                        )
                        
                        # 4. Gripper-only MSE
                        mse_gripper_only = torch.nn.functional.mse_loss(
                            pred_action[..., gripper_idx],
                            gt_action[..., gripper_idx]
                        )
                        
                        # 5. Gripper accuracy (percentage of correct discrete predictions)
                        gt_gripper_discrete = gt_action[..., gripper_idx]
                        gripper_correct = (pred_gripper_discrete == gt_gripper_discrete).float()
                        gripper_accuracy = gripper_correct.mean()
                        
                        # Print metrics
                        print(f"\nTrain Action Metrics (Epoch {self.epoch}) - {action_dim}D (Arm + Gripper):")
                        print(f"  Overall MSE:           {mse_overall.item():.6f}")
                        print(f"  MSE (no gripper):      {mse_no_gripper.item():.6f}")
                        print(f"  MSE (binarized grip):  {mse_binarized.item():.6f}")
                        print(f"  MSE (gripper only):    {mse_gripper_only.item():.6f}")
                        print(f"  Gripper Accuracy:      {gripper_accuracy.item():.4f} ({100*gripper_accuracy.item():.2f}%)")
                        
                        # Log to wandb
                        step_log["train_action_mse_overall"] = mse_overall.item()
                        step_log["train_action_mse_no_gripper"] = mse_no_gripper.item()
                        step_log["train_action_mse_binarized_gripper"] = mse_binarized.item()
                        step_log["train_action_mse_gripper_only"] = mse_gripper_only.item()
                        step_log["train_gripper_accuracy"] = gripper_accuracy.item()
                        step_log["train_action_mse_error"] = mse_overall.item()  # Legacy
                    
                    del batch
                    del obs_dict
                    del gt_action
                    del result
                    del pred_action
                    del mse_overall

            if env_runner is None:
                step_log["test_mean_score"] = -train_loss

            # checkpoint
            if (
                self.epoch % cfg.training.checkpoint_every
            ) == 0 and cfg.checkpoint.save_ckpt:
                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace("/", "_")
                    metric_dict[new_key] = value

                # Handle missing test_mean_score
                if 'test_mean_score' not in metric_dict:
                    if 'val_loss' in metric_dict:
                        metric_dict['test_mean_score'] = -metric_dict['val_loss']
                    else:
                        metric_dict['test_mean_score'] = -train_loss

                # Save periodic checkpoints
                if (self.epoch % 50 == 0):
                    periodic_ckpt_path = os.path.join(self.output_dir, "checkpoints", f"epoch={self.epoch:04d}.ckpt")
                    self.save_checkpoint(path=periodic_ckpt_path)

                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path)
            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log


    # def eval(self):
    #     # load the latest checkpoint
    #     print("Hellooooooo")
    #     cfg = copy.deepcopy(self.cfg)
        
    #     # print(cfg)
    #     # lastest_ckpt_path = self.get_checkpoint_path(tag="latest")
    #     # if lastest_ckpt_path.is_file():
    #         # cprint(f"Resuming from checkpoint {lastest_ckpt_path}", "magenta")
    #         # self.load_checkpoint(path=lastest_ckpt_path)
    #     #  
    #     # lastest_ckpt_path = "/scratch2/cross-emb/DP3_outputs/pybullet_pick_place-dp3-no_eef_seed0/checkpoints"
    #     # lastest_ckpt_path = "/home/aniruth/Desktop/RRC/3D-Diffusion-Policy/checkpoints/latest.ckpt"
    #     lastest_ckpt_path = "/home/aniruth/Desktop/RRC/3D-Diffusion-Policy/checkpoints/latest.ckpt"
    #     print(f"Checkpoint is loaded from : {lastest_ckpt_path}")
        
    #     self.load_checkpoint(path=lastest_ckpt_path)

    #     dataset: BaseDataset
    #     dataset = hydra.utils.instantiate(cfg.task.dataset)

    #     assert isinstance(dataset, BaseDataset), print(
    #         f"dataset must be BaseDataset, got {type(dataset)}"
    #     )

    #     train_dataloader = DataLoader(dataset, **cfg.dataloader)
    #     normalizer = dataset.get_normalizer()

    #     # configure validation dataset
    #     val_dataset = dataset.get_validation_dataset()
    #     val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        
    #     self.model.set_normalizer(normalizer)
    #     if cfg.training.use_ema:
    #         self.ema_model.set_normalizer(normalizer)
        
        
    #     policy = self.model
    #     if cfg.training.use_ema:
    #         policy = self.ema_model
    #         print("Using EMA model for evaluation")


    #     policy.eval()
    #     policy.cuda()

    #     # DEBUG: Verify normalizer is set correctly
    #     cprint("=" * 50, "yellow")
    #     cprint("NORMALIZER KEYS:", "yellow")
    #     cprint(f"{list(policy.normalizer.params_dict.keys())}", "yellow")
    #     # cprint("=" * 50, "yellow")

    #     # Configure environment runner
    #     env_runner: BaseRunner = hydra.utils.instantiate(
    #         cfg.task.env_runner, output_dir=self.output_dir
    #     )
    #     assert isinstance(env_runner, BaseRunner)



    #     cprint(f"Running evaluation with policy...", "green")
    #     runner_log = env_runner.run(policy , dataset = dataset)

    #     cprint("=" * 50, "magenta")
    #     cprint("EVALUATION RESULTS:", "magenta")
    #     for key, value in runner_log.items():
    #         if isinstance(value, (int, float)):
    #             cprint(f"  {key}: {value:.4f}", "magenta")
    #     cprint("=" * 50, "magenta")

    #     return runner_log

    def eval(self):
        # load the latest checkpoint
        print("Hellooooooo")
        cfg = copy.deepcopy(self.cfg)
        
        lastest_ckpt_path = "/home/varun-edachali/Research/RRC/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/outputs/epoch=0950.ckpt"
        print(f"Checkpoint is loaded from : {lastest_ckpt_path}")
        
        self.load_checkpoint(path=lastest_ckpt_path)

        # Load dataset
        dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseDataset), f"dataset must be BaseDataset, got {type(dataset)}"

        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # Configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # Set normalizer
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)
        
        # Select policy
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
            print("Using EMA model for evaluation")

        policy.eval()
        policy.cuda()

        # Verify normalizer
        cprint("=" * 50, "yellow")
        cprint("NORMALIZER KEYS:", "yellow")
        cprint(f"{list(policy.normalizer.params_dict.keys())}", "yellow")

        # Configure environment runner
        env_runner: BaseRunner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=self.output_dir
        )
        assert isinstance(env_runner, BaseRunner)

        # ========== VALIDATION: GT vs PRED ACTIONS ==========
        cprint("=" * 50, "magenta")
        cprint("Running validation to collect GT vs Pred actions...", "magenta")
        cprint("=" * 50, "magenta")
        
        device = torch.device(cfg.training.device)
        all_gt_actions = []
        all_pred_actions = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm.tqdm(val_dataloader, desc="Validation")):
                # Move batch to device
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                
                obs_dict = batch["obs"]
                gt_action = batch["action"]  # Shape: (B, T, action_dim)
                
                # Get predictions
                result = policy.predict_action(obs_dict)
                pred_action = result["action_pred"]  # Shape: (B, T, action_dim)
                
                # Move to CPU and convert to numpy
                gt_action_np = gt_action.cpu().numpy()
                pred_action_np = pred_action.cpu().numpy()
                
                # Store actions
                all_gt_actions.append(gt_action_np)
                all_pred_actions.append(pred_action_np)
        
        # Concatenate all batches
        all_gt_actions = np.concatenate(all_gt_actions, axis=0)  # (N, T, action_dim)
        all_pred_actions = np.concatenate(all_pred_actions, axis=0)  # (N, T, action_dim)
        
        # Create output directory
        action_comparison_dir = os.path.join(self.output_dir, "action_comparisons")
        os.makedirs(action_comparison_dir, exist_ok=True)
        
        # Save with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        gt_actions_path = os.path.join(action_comparison_dir, f"gt_actions_{timestamp}.txt")
        pred_actions_path = os.path.join(action_comparison_dir, f"pred_actions_{timestamp}.txt")
        
        # Save arrays
        np.savetxt(gt_actions_path, all_gt_actions.reshape(-1, all_gt_actions.shape[-1]), 
                fmt='%.6f', header=f'Shape: {all_gt_actions.shape}')
        np.savetxt(pred_actions_path, all_pred_actions.reshape(-1, all_pred_actions.shape[-1]), 
                fmt='%.6f', header=f'Shape: {all_pred_actions.shape}')
        
        # Calculate statistics
        mse_per_dim = np.mean((all_gt_actions - all_pred_actions) ** 2, axis=(0, 1))
        mae_per_dim = np.mean(np.abs(all_gt_actions - all_pred_actions), axis=(0, 1))
        overall_mse = np.mean((all_gt_actions - all_pred_actions) ** 2)
        overall_mae = np.mean(np.abs(all_gt_actions - all_pred_actions))
        
        stats_path = os.path.join(action_comparison_dir, f"action_stats_{timestamp}.txt")
        with open(stats_path, 'w') as f:
            f.write(f"Validation Action Statistics\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"GT Actions Shape: {all_gt_actions.shape}\n")
            f.write(f"Pred Actions Shape: {all_pred_actions.shape}\n")
            f.write(f"\nOverall MSE: {overall_mse:.6f}\n")
            f.write(f"Overall MAE: {overall_mae:.6f}\n")
            f.write(f"\nPer-dimension MSE:\n")
            for i, mse in enumerate(mse_per_dim):
                f.write(f"  Dimension {i}: {mse:.6f}\n")
            f.write(f"\nPer-dimension MAE:\n")
            for i, mae in enumerate(mae_per_dim):
                f.write(f"  Dimension {i}: {mae:.6f}\n")
        
        cprint("=" * 50, "green")
        cprint(f"Saved GT actions to: {gt_actions_path}", "green")
        cprint(f"Saved Pred actions to: {pred_actions_path}", "green")
        cprint(f"Saved statistics to: {stats_path}", "green")
        cprint(f"Overall MSE: {overall_mse:.6f}", "green")
        cprint(f"Overall MAE: {overall_mae:.6f}", "green")
        cprint("=" * 50, "green")
        
        # Run environment evaluation
        cprint(f"Running evaluation with policy...", "green")
        runner_log = env_runner.run(policy, dataset=dataset)

        # Print results
        cprint("=" * 50, "magenta")
        cprint("EVALUATION RESULTS:", "magenta")
        for key, value in runner_log.items():
            if isinstance(value, (int, float)):
                cprint(f"  {key}: {value:.4f}", "magenta")
        cprint("=" * 50, "magenta")

        return runner_log


    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def save_checkpoint(
        self,
        path=None,
        tag="latest",
        exclude_keys=None,
        include_keys=None,
        use_thread=False,
    ):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {"cfg": self.cfg, "state_dicts": dict(), "pickles": dict()}

        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload["state_dicts"][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload["state_dicts"][key] = value.state_dict()
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open("wb"), pickle_module=dill)
            )
            self._saving_thread.start()
        else:
            torch.save(payload, path.open("wb"), pickle_module=dill)

        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())

    def get_checkpoint_path(self, tag="latest"):
        if tag == "latest":
            return pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        elif tag == "best":
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath("checkpoints")
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if "latest" in ckpt:
                    continue
                score = float(ckpt.split("test_mean_score=")[1].split(".ckpt")[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath("checkpoints", best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload["pickles"].keys()

        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    def load_checkpoint(
        self, path=None, tag="latest", exclude_keys=None, include_keys=None, **kwargs
    ):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload

    @classmethod
    def create_from_checkpoint(
        cls, path, exclude_keys=None, include_keys=None, **kwargs
    ):
        payload = torch.load(open(path, "rb"), pickle_module=dill)
        instance = cls(payload["cfg"])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs,
        )
        return instance

    def save_snapshot(self, tag="latest"):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath("snapshots", f"{tag}.pkl")
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, "rb"), pickle_module=dill)


@hydra.main(
    version_base=None,
    config_path=str(
        pathlib.Path(__file__).parent.joinpath("diffusion_policy_3d", "config")
    ),
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
