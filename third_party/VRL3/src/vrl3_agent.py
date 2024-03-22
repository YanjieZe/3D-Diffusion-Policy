# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from transfer_util import initialize_model
from stage1_models import BasicBlock, ResNet84
import os
import copy
from PIL import Image
import platform
from numbers import Number
import utils

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class Identity(nn.Module):
    def __init__(self, input_placeholder=None):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class RLEncoder(nn.Module):
    def __init__(self, obs_shape, model_name, device):
        super().__init__()
        # a wrapper over a non-RL encoder model
        self.device = device
        assert len(obs_shape) == 3
        self.n_input_channel = obs_shape[0]
        assert self.n_input_channel % 3 == 0
        self.n_images = self.n_input_channel // 3
        self.model = self.init_model(model_name)
        self.model.fc = Identity()
        self.repr_dim = self.model.get_feature_size()

        self.normalize_op = transforms.Normalize((0.485, 0.456, 0.406),
                                                 (0.229, 0.224, 0.225))
        self.channel_mismatch = True

    def init_model(self, model_name):
        # model name is e.g. resnet6_32channel
        n_layer_string, n_channel_string = model_name.split('_')
        layer_string_to_layer_list = {
            'resnet6': [0, 0, 0, 0],
            'resnet10': [1, 1, 1, 1],
            'resnet18': [2, 2, 2, 2],
        }
        channel_string_to_n_channel = {
            '32channel': 32,
            '64channel': 64,
        }
        layer_list = layer_string_to_layer_list[n_layer_string]
        start_num_channel = channel_string_to_n_channel[n_channel_string]
        return ResNet84(BasicBlock, layer_list, start_num_channel=start_num_channel).to(self.device)

    def expand_first_layer(self):
        # convolutional channel expansion to deal with input mismatch
        multiplier = self.n_images
        self.model.conv1.weight.data = self.model.conv1.weight.data.repeat(1,multiplier,1,1) / multiplier
        means = (0.485, 0.456, 0.406) * multiplier
        stds = (0.229, 0.224, 0.225) * multiplier
        self.normalize_op = transforms.Normalize(means, stds)
        self.channel_mismatch = False

    def freeze_bn(self):
        # freeze batch norm layers (VRL3 ablation shows modifying how
        # batch norm is trained does not affect performance)
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

    def get_parameters_that_require_grad(self):
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                params.append(param)
        return params

    def transform_obs_tensor_batch(self, obs):
        # transform obs batch before put into the pretrained resnet
        new_obs = self.normalize_op(obs.float()/255)
        return new_obs

    def _forward_impl(self, x):
        x = self.model.get_features(x)
        return x

    def forward(self, obs):
        o = self.transform_obs_tensor_batch(obs)
        h = self._forward_impl(o)
        return h

class Stage3ShallowEncoder(nn.Module):
    def __init__(self, obs_shape, n_channel):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = n_channel * 35 * 35

        self.n_input_channel = obs_shape[0]
        self.conv1 = nn.Conv2d(obs_shape[0], n_channel, 3, stride=2)
        self.conv2 = nn.Conv2d(n_channel, n_channel, 3, stride=1)
        self.conv3 = nn.Conv2d(n_channel, n_channel, 3, stride=1)
        self.conv4 = nn.Conv2d(n_channel, n_channel, 3, stride=1)
        self.relu = nn.ReLU(inplace=True)

        # TODO here add prediction head so we can do contrastive learning...

        self.apply(utils.weight_init)
        self.normalize_op = transforms.Normalize((0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
                                                 (0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225))

        self.compress = nn.Sequential(nn.Linear(self.repr_dim, 50), nn.LayerNorm(50), nn.Tanh())
        self.pred_layer = nn.Linear(50, 50, bias=False)

    def transform_obs_tensor_batch(self, obs):
        # transform obs batch before put into the pretrained resnet
        # correct order might be first augment, then resize, then normalize
        # obs = F.interpolate(obs, size=self.pretrained_model_input_size)
        new_obs = obs / 255.0 - 0.5
        # new_obs = self.normalize_op(new_obs)
        return new_obs

    def _forward_impl(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        return x

    def forward(self, obs):
        o = self.transform_obs_tensor_batch(obs)
        h = self._forward_impl(o)
        h = h.view(h.shape[0], -1)
        return h

    def get_anchor_output(self, obs, actions=None):
        # typically go through conv and then compression layer and then a mlp
        # used for UL update
        conv_out = self.forward(obs)
        compressed = self.compress(conv_out)
        pred = self.pred_layer(compressed)
        return pred, conv_out

    def get_positive_output(self, obs):
        # typically go through conv, compression
        # used for UL update
        conv_out = self.forward(obs)
        compressed = self.compress(conv_out)
        return compressed

class Encoder(nn.Module):
    def __init__(self, obs_shape, n_channel):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = n_channel * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], n_channel, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(n_channel, n_channel, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(n_channel, n_channel, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(n_channel, n_channel, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 1
        self.repr_dim = obs_shape[0]

    def forward(self, obs):
        return obs

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.action_shift=0
        self.action_scale=1
        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        mu = mu * self.action_scale + self.action_shift
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

    def forward_with_pretanh(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        pretanh = mu
        mu = torch.tanh(mu)
        mu = mu * self.action_scale + self.action_shift
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist, pretanh

class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2

class VRL3Agent:
    def __init__(self, obs_shape, action_shape, device, use_sensor, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_clip, use_tb, use_data_aug, encoder_lr_scale,
                 stage1_model_name, safe_q_target_factor, safe_q_threshold, pretanh_penalty, pretanh_threshold,
                 stage2_update_encoder, cql_weight, cql_temp, cql_n_random, stage2_std, stage2_bc_weight,
                 stage3_update_encoder, std0, std1, std_n_decay,
                 stage3_bc_lam0, stage3_bc_lam1):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps

        self.stage2_std = stage2_std
        self.stage2_update_encoder = stage2_update_encoder

        if std1 > std0:
            std1 = std0
        self.stddev_schedule = "linear(%s,%s,%s)" % (str(std0), str(std1), str(std_n_decay))

        self.stddev_clip = stddev_clip
        self.use_data_aug = use_data_aug
        self.safe_q_target_factor = safe_q_target_factor
        self.q_threshold = safe_q_threshold
        self.pretanh_penalty = pretanh_penalty

        self.cql_temp = cql_temp
        self.cql_weight = cql_weight
        self.cql_n_random = cql_n_random

        self.pretanh_threshold = pretanh_threshold

        self.stage2_bc_weight = stage2_bc_weight
        self.stage3_bc_lam0 = stage3_bc_lam0
        self.stage3_bc_lam1 = stage3_bc_lam1

        if stage3_update_encoder and encoder_lr_scale > 0 and len(obs_shape) > 1:
            self.stage3_update_encoder = True
        else:
            self.stage3_update_encoder = False

        self.encoder = RLEncoder(obs_shape, stage1_model_name, device).to(device)

        self.act_dim = action_shape[0]

        if use_sensor:
            downstream_input_dim = self.encoder.repr_dim + 24
        else:
            downstream_input_dim = self.encoder.repr_dim

        self.actor = Actor(downstream_input_dim, action_shape, feature_dim,
                           hidden_dim).to(device)
        self.critic = Critic(downstream_input_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(downstream_input_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        encoder_lr = lr * encoder_lr_scale
        """ set up encoder optimizer """
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)
        self.train()
        self.critic_target.train()

    def load_pretrained_encoder(self, model_path, verbose=True):
        if verbose:
            print("Trying to load pretrained model from:", model_path)
        checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        state_dict = checkpoint['state_dict']

        pretrained_dict = {}
        # remove `module.` if model was pretrained with distributed mode
        for k, v in state_dict.items():
            if 'module.' in k:
                name = k[7:]
            else:
                name = k
            pretrained_dict[name] = v
        self.encoder.model.load_state_dict(pretrained_dict, strict=False)
        if verbose:
            print("Pretrained model loaded!")

    def switch_to_RL_stages(self, verbose=True):
        # run convolutional channel expansion to match input shape
        self.encoder.expand_first_layer()
        if verbose:
            print("Convolutional channel expansion finished: now can take in %d images as input." % self.encoder.n_images)

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode, obs_sensor=None, is_tensor_input=False, force_action_std=None):
        """
        obs: 3x84x84, uint8, [0,255]
        """
        # eval_mode should be False when taking an exploration action in stage 3
        # eval_mode should be True when evaluate agent performance

        if force_action_std == None:
            stddev = utils.schedule(self.stddev_schedule, step)
            if step < self.num_expl_steps and not eval_mode:
                action = np.random.uniform(0, 1, (self.act_dim,)).astype(np.float32)
                return action
        else:
            stddev = force_action_std

        if is_tensor_input:
            obs = self.encoder(obs)
        else:
            obs = torch.as_tensor(obs, device=self.device)
            obs = self.encoder(obs.unsqueeze(0))

        if obs_sensor is not None:
            obs_sensor = torch.as_tensor(obs_sensor, device=self.device)
            obs_sensor = obs_sensor.unsqueeze(0)
            obs_combined = torch.cat([obs, obs_sensor], dim=1)
        else:
            obs_combined = obs

        dist = self.actor(obs_combined, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update(self, replay_iter, step, stage, use_sensor):
        # for stage 2 and 3, we use the same functions but with different hyperparameters
        assert stage in (2, 3)
        metrics = dict()

        if stage == 2:
            update_encoder = self.stage2_update_encoder
            stddev = self.stage2_std
            conservative_loss_weight = self.cql_weight
            bc_weight = self.stage2_bc_weight

        if stage == 3:
            if step % self.update_every_steps != 0:
                return metrics
            update_encoder = self.stage3_update_encoder

            stddev = utils.schedule(self.stddev_schedule, step)
            conservative_loss_weight = 0

            # compute stage 3 BC weight
            bc_data_per_iter = 40000
            i_iter = step // bc_data_per_iter
            bc_weight = self.stage3_bc_lam0 * self.stage3_bc_lam1 ** i_iter

        # batch data
        batch = next(replay_iter)
        if use_sensor: # TODO might want to...?
            obs, action, reward, discount, next_obs, obs_sensor, obs_sensor_next = utils.to_torch(batch, self.device)
        else:
            obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)
            obs_sensor, obs_sensor_next = None, None

        # augment
        if self.use_data_aug:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
        else:
            obs = obs.float()
            next_obs = next_obs.float()

        # encode
        if update_encoder:
            obs = self.encoder(obs)
        else:
            with torch.no_grad():
                obs = self.encoder(obs)

        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # concatenate obs with additional sensor observation if needed
        obs_combined = torch.cat([obs, obs_sensor], dim=1) if obs_sensor is not None else obs
        obs_next_combined = torch.cat([next_obs, obs_sensor_next], dim=1) if obs_sensor_next is not None else next_obs

        # update critic
        metrics.update(self.update_critic_vrl3(obs_combined, action, reward, discount, obs_next_combined,
                                               stddev, update_encoder, conservative_loss_weight))

        # update actor, following previous works, we do not use actor gradient for encoder update
        metrics.update(self.update_actor_vrl3(obs_combined.detach(), action, stddev, bc_weight,
                                              self.pretanh_penalty, self.pretanh_threshold))

        metrics['batch_reward'] = reward.mean().item()

        # update critic target networks
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)
        return metrics

    def update_critic_vrl3(self, obs, action, reward, discount, next_obs, stddev, update_encoder, conservative_loss_weight):
        metrics = dict()
        batch_size = obs.shape[0]

        """
        STANDARD Q LOSS COMPUTATION:
        - get standard Q loss first, this is the same as in any other online RL methods
        - except for the safe Q technique, which controls how large the Q value can be
        """
        with torch.no_grad():
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

            if self.safe_q_target_factor < 1:
                target_Q[target_Q > (self.q_threshold + 1)] = self.q_threshold + (target_Q[target_Q > (self.q_threshold+1)] - self.q_threshold) ** self.safe_q_target_factor

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        """
        CONSERVATIVE Q LOSS COMPUTATION:
        - sample random actions, actions from policy and next actions from policy, as done in CQL authors' code
          (though this detail is not really discussed in the CQL paper)
        - only compute this loss when conservative loss weight > 0
        """
        if conservative_loss_weight > 0:
            random_actions = (torch.rand((batch_size * self.cql_n_random, self.act_dim), device=self.device) - 0.5) * 2

            dist = self.actor(obs, stddev)
            current_actions = dist.sample(clip=self.stddev_clip)

            dist = self.actor(next_obs, stddev)
            next_current_actions = dist.sample(clip=self.stddev_clip)

            # now get Q values for all these actions (for both Q networks)
            obs_repeat = obs.unsqueeze(1).repeat(1, self.cql_n_random, 1).view(obs.shape[0] * self.cql_n_random,
                                                                               obs.shape[1])

            Q1_rand, Q2_rand = self.critic(obs_repeat,
                                           random_actions)  # TODO might want to double check the logic here see if the repeat is correct
            Q1_rand = Q1_rand.view(obs.shape[0], self.cql_n_random)
            Q2_rand = Q2_rand.view(obs.shape[0], self.cql_n_random)

            Q1_curr, Q2_curr = self.critic(obs, current_actions)
            Q1_curr_next, Q2_curr_next = self.critic(obs, next_current_actions)

            # now concat all these Q values together
            Q1_cat = torch.cat([Q1_rand, Q1, Q1_curr, Q1_curr_next], 1)
            Q2_cat = torch.cat([Q2_rand, Q2, Q2_curr, Q2_curr_next], 1)

            cql_min_q1_loss = torch.logsumexp(Q1_cat / self.cql_temp,
                                              dim=1, ).mean() * conservative_loss_weight * self.cql_temp
            cql_min_q2_loss = torch.logsumexp(Q2_cat / self.cql_temp,
                                              dim=1, ).mean() * conservative_loss_weight * self.cql_temp

            """Subtract the log likelihood of data"""
            conservative_q_loss = cql_min_q1_loss + cql_min_q2_loss - (Q1.mean() + Q2.mean()) * conservative_loss_weight
            critic_loss_combined = critic_loss + conservative_q_loss
        else:
            critic_loss_combined = critic_loss

        # logging
        metrics['critic_target_q'] = target_Q.mean().item()
        metrics['critic_q1'] = Q1.mean().item()
        metrics['critic_q2'] = Q2.mean().item()
        metrics['critic_loss'] = critic_loss.item()

        # if needed, also update encoder with critic loss
        if update_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss_combined.backward()
        self.critic_opt.step()
        if update_encoder:
            self.encoder_opt.step()

        return metrics

    def update_actor_vrl3(self, obs, action, stddev, bc_weight, pretanh_penalty, pretanh_threshold):
        metrics = dict()

        """
        get standard actor loss
        """
        dist, pretanh = self.actor.forward_with_pretanh(obs, stddev)
        current_action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(current_action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, current_action)
        Q = torch.min(Q1, Q2)
        actor_loss = -Q.mean()

        """
        add BC loss
        """
        if bc_weight > 0:
            # get mean action with no action noise (though this might not be necessary)
            stddev_bc = 0
            dist_bc = self.actor(obs, stddev_bc)
            current_mean_action = dist_bc.sample(clip=self.stddev_clip)
            actor_loss_bc = F.mse_loss(current_mean_action, action) * bc_weight
        else:
            actor_loss_bc = torch.FloatTensor([0]).to(self.device)

        """
        add pretanh penalty (might not be necessary for Adroit)
        """
        pretanh_loss = 0
        if pretanh_penalty > 0:
            pretanh_loss = pretanh.abs() - pretanh_threshold
            pretanh_loss[pretanh_loss < 0] = 0
            pretanh_loss = (pretanh_loss ** 2).mean() * pretanh_penalty

        """
        combine actor losses and optimize
        """
        actor_loss_combined = actor_loss + actor_loss_bc + pretanh_loss

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss_combined.backward()
        self.actor_opt.step()

        metrics['actor_loss'] = actor_loss.item()
        metrics['actor_loss_bc'] = actor_loss_bc.item()
        metrics['actor_logprob'] = log_prob.mean().item()
        metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
        metrics['abs_pretanh'] = pretanh.abs().mean().item()
        metrics['max_abs_pretanh'] = pretanh.abs().max().item()

        return metrics

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.encoder.to(device)
        self.device = device