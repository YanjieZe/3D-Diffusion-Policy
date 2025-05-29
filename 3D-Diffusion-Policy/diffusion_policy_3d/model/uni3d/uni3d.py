import torch
import timm
import numpy as np
import os
import yaml
from torch import nn
import logging

from .point_encoder import PointcloudEncoder

class Uni3D(nn.Module):
    def __init__(self, point_encoder, output_dim=None, load_pretrain=None):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.point_encoder = point_encoder
        self.load_pretrain = load_pretrain
        # Add optional projection layer to map from 1024 to desired output dimension
        self.output_dim = output_dim
        if output_dim is not None and output_dim != 1024:
            self.projection = nn.Linear(1024, output_dim)
            logging.info(f'Added projection layer: 1024 -> {output_dim}')
        else:
            self.projection = None

    def forward(self, pc):
        if self.load_pretrain:  
            pc = self.uni3d_pcd_mapping(pc)
            
        xyz = pc[:,:,:3].contiguous() # B N 3
        color = pc[:,:,3:].contiguous() # B N 3
        pc_feat = self.point_encoder(xyz, color)
        
        # Apply projection if specified
        if self.projection is not None:
            pc_feat = self.projection(pc_feat)
            
        return pc_feat
    
    def uni3d_pcd_mapping(self, pcd):
        # HACK: DP3 uses normalizer to normalize the point cloud to -1~1 like other inputs,
        # but the point cloud that uni3d takes during pre-training is in the range of 0~1
        # so we need to map the point cloud to the range of 0~1
        assert len(pcd.shape) == 3, f"pcd shape must be B, N, 3, but got {pcd.shape}"
        assert pcd.min().item() == -1.0, f"Point cloud minimum value {pcd.min().item()} should be -1.0"
        assert pcd.max().item() == 1.0, f"Point cloud maximum value {pcd.max().item()} should be 1.0"
        return (pcd + 1.0) / 2.0

def create_uni3d(args, output_dim=None):  
    # Assert that freeze_weights cannot be true when load_pretrain is false
    if not getattr(args, 'load_pretrain', True):
        assert not getattr(args, 'freeze_weights', False), "Cannot freeze weights when load_pretrain is False"

    uni3d_size_args = get_uni3d_size_args(args.uni3d_size)
    args["pc_feat_dim"] = uni3d_size_args["pc_feat_dim"]
    
    # create transformer blocks for point cloud via timm
    drop_path_rate = 0.0 if args.freeze_weights else args.drop_path_rate
    point_transformer = timm.create_model(uni3d_size_args["pc_model"], checkpoint_path=args.pretrained_pc, drop_path_rate=drop_path_rate)

    # create whole point cloud encoder
    point_encoder = PointcloudEncoder(point_transformer, args)

    # uni3d model with optional output dimension
    model = Uni3D(point_encoder=point_encoder, output_dim=output_dim, load_pretrain=args.load_pretrain)
    
    # Only load pretrained weights if load_pretrain is True
    if getattr(args, 'load_pretrain', True):
        assert hasattr(args, 'checkpoint_path') and args.checkpoint_path is not None, "checkpoint_path is required when load_pretrain is True"
        checkpoint_path = args.checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logging.info('Loading Uni3D checkpoint from {}'.format(checkpoint_path))
        
        # Extract state dict based on the format in the checkpoint
        if 'module' in checkpoint:
            sd = checkpoint['module']
        elif 'state_dict' in checkpoint:
            sd = checkpoint['state_dict']
        else:
            sd = checkpoint
        
        # Remove 'module.' prefix if necessary
        distributed = getattr(args, 'distributed', False)
        if not distributed and next(iter(sd.items()))[0].startswith('module.'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        
        # Load the state dict (excluding projection layer if it exists)
        model_state_dict = model.state_dict()
        
        # Filter out projection layer from both model and checkpoint
        filtered_sd = {}
        filtered_model_keys = set()
        
        for key, value in sd.items():
            if not key.startswith('projection.'):
                filtered_sd[key] = value
        
        for key in model_state_dict.keys():
            if not key.startswith('projection.'):
                filtered_model_keys.add(key)
        
        # Check which parameters are in the model but not in the checkpoint (missing)
        checkpoint_keys = set(filtered_sd.keys())
        
        missing_keys = filtered_model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - filtered_model_keys
        
        # Log detailed information about the loading process
        if len(missing_keys) > 0:
            logging.info(f"Missing keys in checkpoint: {len(missing_keys)} parameters")
            for key in sorted(missing_keys):
                logging.info(f"  Missing: {key}")
        
        if len(unexpected_keys) > 0:
            logging.info(f"Unexpected keys in checkpoint: {len(unexpected_keys)} parameters")
            for key in sorted(unexpected_keys):
                logging.info(f"  Unexpected: {key}")
                
        matched_keys = filtered_model_keys.intersection(checkpoint_keys)
        logging.info(f"Successfully matched keys: {len(matched_keys)}/{len(filtered_model_keys)} model parameters")
        
        # Load the state dict (strict=False to allow missing projection layer)
        model.load_state_dict(filtered_sd, strict=False)
        logging.info('Successfully loaded Uni3D model weights')
    else:
        logging.info('Skipping pretrained weights loading as load_pretrain is False')
    
    # Freeze extractor if requested
    if args.freeze_weights:
        model.eval()
        logging.info('Freezing Uni3D model parameters')
        
        # Freeze all parameters except projection layer
        for name, param in model.named_parameters():
            if not name.startswith('projection.'):
                param.requires_grad = False
        
        # Convert to half precision when frozen, but keep projection layer in full precision
        for name, module in model.named_modules():
            if not name.startswith('projection') and hasattr(module, 'half'):
                module.half()
        
        logging.info('Converted Uni3D model to half precision (except projection layer)')
        
        # Log projection layer status
        if hasattr(model, 'projection') and model.projection is not None:
            logging.info('Projection layer remains trainable and in full precision')
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    point_encoder_params = sum(p.numel() for p in model.point_encoder.parameters())
    projection_params = sum(p.numel() for p in model.projection.parameters()) if hasattr(model, 'projection') and model.projection is not None else 0
    
    logging.info(f"Model parameters: {total_params:,} total")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    logging.info(f"Point encoder parameters: {point_encoder_params:,}")
    if projection_params > 0:
        logging.info(f"Projection layer parameters: {projection_params:,}")
    
    return model

def get_uni3d_size_args(uni3d_size):
    if uni3d_size == "base":
        pc_model = "eva02_base_patch14_448"
        pc_feat_dim = 768
    elif uni3d_size == "large":
        pc_model = "eva02_large_patch14_448"
        pc_feat_dim = 1024
    elif uni3d_size == "giant":
        pc_model = "eva_giant_patch14_560"
        pc_feat_dim = 1408
    elif uni3d_size == "tiny":
        pc_model = "eva02_tiny_patch14_224"
        pc_feat_dim = 192
    elif uni3d_size == "small":
        pc_model="eva02_small_patch14_224"
        pc_feat_dim=384
    return {
        "pc_model": pc_model,
        "pc_feat_dim": pc_feat_dim,
    }


# Example of loading a pretrained Uni3D model for downstream tasks:
"""
Example usage:

import argparse
from models.uni3d import create_uni3d

# Create arguments
parser = argparse.ArgumentParser()
# Add required arguments for model creation
parser.add_argument('--pc_model', type=str, default='eva_giant_patch14_560', help='Point cloud backbone model name')
parser.add_argument('--pretrained_pc', type=str, default=None, help='Path to backbone pretrained weights (optional)')
parser.add_argument('--drop_path_rate', type=float, default=0.0, help='Drop path rate')
parser.add_argument('--distributed', action='store_true', help='Whether using distributed training')
# Add other required arguments from your configuration
parser.add_argument('--pc_feat_dim', type=int, default=768, help='Point cloud feature dimension')
parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
parser.add_argument('--group_size', type=int, default=32, help='Group size')
parser.add_argument('--num_group', type=int, default=512, help='Number of groups')
parser.add_argument('--pc_encoder_dim', type=int, default=256, help='Point encoder dimension')
parser.add_argument('--patch_dropout', type=float, default=0.0, help='Patch dropout rate')

args = parser.parse_args()

# Path to the pretrained Uni3D checkpoint
checkpoint_path = 'path/to/uni3d_checkpoints/uni3d-g/model.pt'

# Create model with pretrained weights
model = create_uni3d(args)

# Now you can use the model for downstream tasks
# model.eval()  # Set to evaluation mode for inference
# pc_features = model.encode_pc(point_cloud)  # Extract point cloud features
"""


