import os, pdb
import numpy as np
import math, torch
from torch import nn
from ..ocl.perceptual_grouping import build_mlp

def get_norm(norm, out_channels):
    
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {"GN": lambda channels: nn.GroupNorm(32, channels),
                "LN": lambda channels: nn.GroupNorm(1, channels),}[norm]
    return norm(out_channels)

class FPN(nn.Module):

    def __init__(self, in_channels, slot_dim, slot_channels, fpn_names, scale_factors = [4, 2, 1, 0.5], 
                 strides = [4, 8, 16, 32], out_channels = 512, norm = 'LN', use_bias = False, ):
        super().__init__()
        self.stages = []
        dim = in_channels
        # FPN: stride 16 --> [stride 4, stride 8, stride 16, stride 32]
        for idx, scale in enumerate(scale_factors):
            out_dim = dim // 2
            if scale == 2:
                layers = [nn.ConvTranspose2d(dim, out_dim, kernel_size=2, stride=2),
                          get_norm(norm, out_dim), nn.GELU(),
                          nn.Conv2d(out_dim, out_dim, 1, 1, 0)]
            elif scale == 1:
                layers = [nn.Conv2d(dim, out_dim, 1, 1, 0)]
            elif scale == 0.5:
                layers = [nn.Conv2d(dim, out_dim, 3, 2, 1)]
            elif scale == 0.25:
                layers = [nn.Conv2d(dim, out_dim, 3, 2, 1),
                          get_norm(norm, out_dim), nn.GELU(),
                          nn.Conv2d(out_dim, out_dim, 3, 2, 1)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
            
            
            layers.append(nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=use_bias,))
            layers.append(get_norm(norm, out_dim))
            layers.append(nn.GELU(),)
            layers.append(nn.Conv2d(out_dim, out_channels, kernel_size=3, padding=1, bias=use_bias))
            layers.append(get_norm(norm, out_channels))
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simple_fpn_{stage}", layers)
            self.stages.append(layers)

        self.fpn_names = fpn_names
        self.query_proj = build_mlp(slot_dim, out_channels, slot_dim, 2)

    def forward(self, in_features, slots):

        results = []
        for k, stage in enumerate(self.stages):
            results.append(stage(in_features))
        features = {k: res for k, res in zip(self.fpn_names, results)}
        queries = self.query_proj(slots)
        return features, queries

