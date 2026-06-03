from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import pytorch3d.ops as torch3d_ops

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.simple_conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder
from diffusion_policy_3d.policy.adaptive_action import (
    init_adaptive_action_policy,
    reset_adaptive_action_policy,
    select_adaptive_action,
)


class ChunkSizePredictor(nn.Module):
    def __init__(self, action_dim, obs_dim, min_chunk, max_chunk, hidden_dim=128):
        super().__init__()
        assert max_chunk >= min_chunk >= 1
        self.min_chunk = int(min_chunk)
        self.max_chunk = int(max_chunk)
        self.n_options = self.max_chunk - self.min_chunk + 1
        self.net = nn.Sequential(
            nn.Linear(action_dim + obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_options),
        )

    def forward(self, action_seq, obs_feat):
        action_feat = action_seq.mean(dim=1)
        obs_feat = obs_feat.reshape(obs_feat.shape[0], -1)
        return self.net(torch.cat([action_feat, obs_feat], dim=-1))

    def get_chunk_size(self, action_seq, obs_feat):
        logits = self.forward(action_seq, obs_feat)
        indices = torch.argmax(logits, dim=-1)
        return self.min_chunk + indices


class SimpleDP3(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            execution_mode="fixed",
            min_action_steps=1,
            mid_action_steps=None,
            max_action_steps=None,
            low_threshold=0.03,
            high_threshold=0.08,
            overlap_alpha=0.5,
            schedule_boundaries=None,
            schedule_action_steps=None,
            uncertainty_samples=1,
            phase_selector_path=None,
            phase_selector_steps=None,
            temporal_loss_weight=0.0,
            temporal_loss_center=0.55,
            temporal_loss_width=0.10,
            n_overlap=0,
            use_dynamic_chunk_head=False,
            chunk_hidden_dim=128,
            chunk_loss_weight=0.1,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])


        obs_encoder = DP3Encoder(observation_space=obs_dict,
                                                   img_crop_shape=crop_shape,
                                                out_channel=encoder_output_dim,
                                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                use_pc_color=use_pc_color,
                                                pointnet_type=pointnet_type,
                                                )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[SDP3] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[SDP3] pointnet_type: {self.pointnet_type}", "yellow")


        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.temporal_loss_weight = float(temporal_loss_weight)
        self.temporal_loss_center = float(temporal_loss_center)
        self.temporal_loss_width = float(temporal_loss_width)
        self.n_overlap = int(n_overlap)
        self.use_dynamic_chunk_head = bool(use_dynamic_chunk_head)
        self.chunk_loss_weight = float(chunk_loss_weight)
        self.chunk_predictor = None
        if self.use_dynamic_chunk_head:
            assert self.obs_as_global_cond, "dynamic chunk head requires obs_as_global_cond=True"
            self.chunk_predictor = ChunkSizePredictor(
                action_dim=action_dim,
                obs_dim=global_cond_dim,
                min_chunk=n_action_steps,
                max_chunk=horizon - (n_obs_steps - 1),
                hidden_dim=chunk_hidden_dim,
            )
            cprint(
                f"[SDP3] ChunkSizePredictor: chunk_size in "
                f"[{self.chunk_predictor.min_chunk}, {self.chunk_predictor.max_chunk}]",
                "cyan")
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        init_adaptive_action_policy(
            self,
            mode=execution_mode,
            min_action_steps=min_action_steps,
            mid_action_steps=mid_action_steps,
            max_action_steps=max_action_steps,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            overlap_alpha=overlap_alpha,
            schedule_boundaries=schedule_boundaries,
            schedule_action_steps=schedule_action_steps,
            uncertainty_samples=uncertainty_samples,
            phase_selector_path=phase_selector_path,
            phase_selector_steps=phase_selector_steps,
        )


        print_params(self)

    def reset(self):
        reset_adaptive_action_policy(self)
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler


        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)


        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]


            model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond)
            
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor],
                       leftover_actions=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        start = To - 1
        n_committed = 0
        if leftover_actions is not None and self.n_overlap > 0:
            n_committed = min(int(self.n_overlap), leftover_actions.shape[1])
            leftover_actions = leftover_actions.to(device=device, dtype=dtype)
            normalized_leftover = self.normalizer['action'].normalize(
                leftover_actions[:, :n_committed])
            cond_data[:, start:start + n_committed, :Da] = normalized_leftover
            cond_mask[:, start:start + n_committed, :Da] = True

        # run sampling
        if getattr(self, "execution_mode", "fixed") in (
                "uncertainty", "overlap_uncertainty", "best_of_n",
                "best_of_n_uncertainty"):
            samples = []
            for _ in range(int(getattr(self, "uncertainty_samples", 1))):
                samples.append(self.conditional_sample(
                    cond_data,
                    cond_mask,
                    local_cond=local_cond,
                    global_cond=global_cond,
                    **self.kwargs)[..., :Da])
            naction_samples = torch.stack(samples, dim=0)
            uncertainty_profile = torch.linalg.norm(
                naction_samples.std(dim=0, unbiased=False), dim=-1)
            if getattr(self, "execution_mode", "fixed") in (
                    "best_of_n", "best_of_n_uncertainty"):
                sample_mean = naction_samples.mean(dim=0, keepdim=True)
                centrality = torch.linalg.norm(
                    naction_samples - sample_mean, dim=-1).mean(dim=-1)
                velocity = naction_samples[:, :, 1:] - naction_samples[:, :, :-1]
                smoothness = torch.linalg.norm(velocity, dim=-1).mean(dim=-1)
                score = centrality + 0.25 * smoothness
                best_idx = torch.argmin(score, dim=0)
                gather_idx = best_idx.view(1, B, 1, 1).expand(1, B, T, Da)
                naction_pred = torch.gather(naction_samples, 0, gather_idx).squeeze(0)
            else:
                naction_pred = naction_samples.mean(dim=0)
        else:
            nsample = self.conditional_sample(
                cond_data,
                cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                **self.kwargs)
            naction_pred = nsample[..., :Da]
            uncertainty_profile = None

        # unnormalize prediction
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        phase_obs = nobs["agent_pos"][:, To - 1] if "agent_pos" in nobs else None
        action_result = select_adaptive_action(
            self, action_pred, start, uncertainty_profile=uncertainty_profile,
            phase_obs=phase_obs)
        action = action_result['action']

        chunk_size = None
        n_fresh = None
        if self.use_dynamic_chunk_head and self.chunk_predictor is not None:
            full_action = action_pred[:, start:]
            chunk_size_tensor = self.chunk_predictor.get_chunk_size(
                full_action, global_cond)
            chunk_size = int(chunk_size_tensor[0].detach().cpu())
            n_fresh = max(chunk_size - n_committed, 1)
        elif self.n_overlap > 0:
            n_fresh = max(int(self.n_action_steps) - int(self.n_overlap), 1)
        
        # get prediction


        result = {
            'action': action,
            'action_pred': action_pred,
            'action_steps': action_result['action_steps'],
            'adaptive_score': action_result['adaptive_score'],
            'action_start': torch.full(
                (B,), start, device=device, dtype=torch.long),
        }
        if chunk_size is not None:
            result['dynamic_chunk_size'] = torch.full(
                (B,), chunk_size, device=device, dtype=torch.long)
        if n_fresh is not None:
            result['n_fresh'] = torch.full(
                (B,), n_fresh, device=device, dtype=torch.long)
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
       
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()


        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)
        if self.n_overlap > 0 and self.obs_as_global_cond:
            start = self.n_obs_steps - 1
            max_k = min(self.n_overlap, trajectory.shape[1] - start - 1)
            if max_k > 0:
                k = torch.randint(0, max_k + 1, (1,), device=trajectory.device).item()
                if k > 0:
                    condition_mask[:, start:start + k, :self.action_dim] = True

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        


        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        
        pred = self.model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                            local_cond=local_cond, 
                            global_cond=global_cond)


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        if self.temporal_loss_weight > 0 and 'sample_progress' in batch:
            progress = batch['sample_progress'].to(device=loss.device, dtype=loss.dtype)
            progress = progress.reshape(loss.shape[0])
            width = max(self.temporal_loss_width, 1e-6)
            phase_gate = torch.sigmoid((progress - self.temporal_loss_center) / width)
            weights = 1.0 + self.temporal_loss_weight * phase_gate
            loss = loss * weights.unsqueeze(-1)
        loss = loss.mean()
        if self.use_dynamic_chunk_head and self.chunk_predictor is not None:
            action_start = self.n_obs_steps - 1
            action_seq_gt = nactions[:, action_start:]
            with torch.no_grad():
                velocity = action_seq_gt[:, 1:] - action_seq_gt[:, :-1]
                smoothness = torch.linalg.norm(velocity, dim=-1).mean(dim=-1)
                if batch_size > 1:
                    order = torch.argsort(smoothness)
                    ranks = torch.empty_like(order)
                    ranks[order] = torch.arange(batch_size, device=order.device)
                    n_opt = self.chunk_predictor.n_options
                    chunk_label = (
                        (1.0 - ranks.float() / (batch_size - 1 + 1e-6))
                        * (n_opt - 1)
                    ).long().clamp(0, n_opt - 1)
                else:
                    chunk_label = torch.zeros(
                        (batch_size,), device=trajectory.device, dtype=torch.long)
            chunk_logits = self.chunk_predictor(action_seq_gt, global_cond.detach())
            chunk_loss = F.cross_entropy(chunk_logits, chunk_label)
            loss = loss + self.chunk_loss_weight * chunk_loss
        

        loss_dict = {
                'bc_loss': loss.item(),
            }
        if self.use_dynamic_chunk_head and self.chunk_predictor is not None:
            loss_dict['chunk_loss'] = chunk_loss.item()

        # print(f"t2-t1: {t2-t1:.3f}")
        # print(f"t3-t2: {t3-t2:.3f}")
        # print(f"t4-t3: {t4-t3:.3f}")
        # print(f"t5-t4: {t5-t4:.3f}")
        # print(f"t6-t5: {t6-t5:.3f}")
        
        return loss, loss_dict
