import numpy as np
import os, pdb, time
import torch_fidelity
import os.path as osp
from omegaconf import OmegaConf
from paintmind.engine.util import instantiate_from_config

def train_on_coco():

    cfg_file = 'configs/vit_vqgan.yaml'
    assert osp.exists(cfg_file)
    config = OmegaConf.load(cfg_file)
    trainer = instantiate_from_config(config.trainer)
    trainer.train()

if __name__ == '__main__':

    train_on_coco()
