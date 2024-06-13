import numpy as np
import os, math, pdb
import torch.nn as nn
import os.path as osp
from einops import rearrange
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from ..engine.util import instantiate_from_config
from detectron2.projects.deeplab import add_deeplab_config
from ..modules.mask2former.config import add_maskformer2_config
from ..modules.mask2former.mask2former_head import ModifiedMaskFormer

class SegHead(nn.Module):


    def __init__(self, config_file, in_channels, mid_channels, out_channels, origin, fpn_config, unet_config = None):

        super().__init__()
        self.fpn = instantiate_from_config(fpn_config)
        self.seghead = self.build_mask2former(config_file, in_channels, origin)
        self.mid_channels = mid_channels
        self.proj = nn.Sequential(nn.Linear(in_channels, mid_channels),
                                  nn.LayerNorm(mid_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(mid_channels, out_channels),
                                  nn.LayerNorm(out_channels))

    def build_mask2former(self, config_file, in_channels, origin=False):

        # setup config
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.freeze()
        self.mask2former_cfg = cfg

        if (not origin):
            output_shape = {
                "res2": ShapeSpec(channels=in_channels, stride=4),
                "res3": ShapeSpec(channels=in_channels, stride=8),
                "res4": ShapeSpec(channels=in_channels, stride=16),
                "res5": ShapeSpec(channels=in_channels, stride=32),
            }
            config = ModifiedMaskFormer.from_config(cfg, output_shape)
            return ModifiedMaskFormer(**config)

        else:
            config = MaskFormer.from_config(cfg)
            return MaskFormer(**config)
    
    def forward(self, x, slots, img, targets):
        
        h = w = int(math.sqrt(x.size(1)))
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w = w)

        # TODO: build FPN feature
        features, queries = self.fpn(x, slots)

        features, slots, results, outputs = self.seghead(features, queries, img, targets)
        z = rearrange(outputs['multi_scale_features'][2], 'n b c-> b n c')

        z = self.proj(z)
        return z, slots, results, outputs