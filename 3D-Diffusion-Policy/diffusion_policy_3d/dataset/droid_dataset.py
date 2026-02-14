from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from termcolor import cprint


class DroidDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            ):
        super().__init__()
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud', 'img', 'cube_pos'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (T, 13)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 2500, 6)
        cube_pos = sample['cube_pos'][:,].astype(np.float32) # (T, 7) - [x,y,z,qx,qy,qz,qw]

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 2500, 6
                'agent_pos': agent_pos, # T, 13
            },
            'action': sample['action'].astype(np.float32), # T, 13
            'cube_pos': cube_pos, # T, 7 - cube position and orientation
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
    
    def get_episode_cube_start_pos(self, episode_idx: int):
        """Get the initial cube position for a specific episode"""
        # Get the start index of the episode
        episode_start_idx = self.replay_buffer.episode_ends[episode_idx] if episode_idx > 0 else 0
        if episode_idx > 0:
            episode_start_idx = self.replay_buffer.episode_ends[episode_idx - 1]
        
        # Get the first cube position in the episode (initial position)
        cube_pos = self.replay_buffer['cube_pos'][episode_start_idx].astype(np.float32)
        return cube_pos  # Returns [x, y, z, qx, qy, qz, qw]
    
    def get_episode(self, episode_idx: int):
        """
        Get the full trajectory for a specific episode
        
        Args:
            episode_idx: Episode index
            
        Returns:
            Dictionary containing full episode data with keys:
                - 'action': (T, action_dim) - ground truth actions
                - 'state': (T, state_dim) - robot states
                - 'point_cloud': (T, num_points, 6) - point clouds
                - 'cube_pos': (T, 7) - cube positions and orientations
        """
        # Step 1: Find episode boundaries in the replay buffer
        if episode_idx > 0:
            start_idx = self.replay_buffer.episode_ends[episode_idx - 1]
        else:
            start_idx = 0
        end_idx = self.replay_buffer.episode_ends[episode_idx]
        
        # Step 2: Extract the slice of actions for this episode
        episode_data = {
            'action': self.replay_buffer['action'][start_idx:end_idx].astype(np.float32), # This gives us array of shape (350, 7) - all actions in this episode
            'state': self.replay_buffer['state'][start_idx:end_idx].astype(np.float32),
            'point_cloud': self.replay_buffer['point_cloud'][start_idx:end_idx].astype(np.float32),
            'cube_pos': self.replay_buffer['cube_pos'][start_idx:end_idx].astype(np.float32),
        }

        

        return episode_data
        