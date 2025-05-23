import torch
import timm
import numpy as np
import os
import yaml
from torch import nn
import logging

from .point_encoder import PointcloudEncoder

class Uni3D(nn.Module):
    def __init__(self, point_encoder):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.point_encoder = point_encoder

    def forward(self, pc):
        xyz = pc[:,:,:3].contiguous() # B N 3
        color = pc[:,:,3:].contiguous() # B N 3
        pc_feat = self.point_encoder(xyz, color)
        return pc_feat

def create_uni3d(args):  

    uni3d_size_args = get_uni3d_size_args(args.uni3d_size)
    args["pc_feat_dim"] = uni3d_size_args["pc_feat_dim"]
    
    # create transformer blocks for point cloud via timm
    point_transformer = timm.create_model(uni3d_size_args["pc_model"], checkpoint_path=args.pretrained_pc, drop_path_rate=args.drop_path_rate)

    # create whole point cloud encoder
    point_encoder = PointcloudEncoder(point_transformer, args)

    # uni3d model
    model = Uni3D(point_encoder=point_encoder)
    
    # load full Uni3D pretrained weights if checkpoint_path is provided
    assert hasattr(args, 'checkpoint_path') and args.checkpoint_path is not None, "checkpoint_path is required"
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
    
    # Check which parameters are in the model but not in the checkpoint (missing)
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(sd.keys())
    
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    
    # Log detailed information about the loading process
    if len(missing_keys) > 0:
        logging.info(f"Missing keys in checkpoint: {len(missing_keys)} parameters")
        for key in sorted(missing_keys):
            logging.info(f"  Missing: {key}")
    
    if len(unexpected_keys) > 0:
        logging.info(f"Unexpected keys in checkpoint: {len(unexpected_keys)} parameters")
        for key in sorted(unexpected_keys):
            logging.info(f"  Unexpected: {key}")
            
    matched_keys = model_keys.intersection(checkpoint_keys)
    logging.info(f"Successfully matched keys: {len(matched_keys)}/{len(model_keys)} model parameters")
    
    # Load the state dict
    model.load_state_dict(sd, strict=False)
    logging.info('Successfully loaded Uni3D model weights')
    
    # Freeze extractor if requested
    if args.freeze_weights:
        model.eval()
        logging.info('Freezing Uni3D model parameters')
        for param in model.parameters():
            param.requires_grad = False
        # Convert to half precision when frozen
        model = model.half()
        logging.info('Converted Uni3D model to half precision')
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    point_encoder_params = sum(p.numel() for p in model.point_encoder.parameters())
    
    logging.info(f"Model parameters: {total_params:,} total")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    logging.info(f"Point encoder parameters: {point_encoder_params:,}")
    
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


