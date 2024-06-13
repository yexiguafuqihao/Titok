#@title Declare Dependencies
import sys, os
import io, requests
import numpy as np
import torch_fidelity
import os.path as osp
from glob import glob
import paintmind as pm
import torch, gc, cv2, pdb
from omegaconf import OmegaConf
import torchvision.transforms as T
from matplotlib import pyplot as plt
from  torch.utils.data import DataLoader
from torchvision.utils import save_image
from IPython.display import clear_output
from paintmind.utils.datasets import CoCo
from paintmind.factory import create_model
import torchvision.transforms.functional as TF
from paintmind.reconstruct import download_image
from paintmind.reconstruct import reconstruction
from paintmind.utils.trainer import PaintMindTrainer
from paintmind.utils.transform import stage1_transform
from paintmind.reconstruct import restore
from PIL import Image, ImageDraw, ImageFont

def reconstruction2():
    dirpath = '/home/zhenganlin/june/MSCOCO/val2017/'
    images = glob(osp.join(dirpath, '*.jpg'))

    saveDir = 'rec'
    os.makedirs(saveDir, exist_ok=True)
    print(f'{len(images)}')
    for i, img_path in enumerate(images):

        if not osp.exists(img_path):
          continue
          
        checkpoint_path = 'output/models/vit_vq_step_69000.pt'
        fig = reconstruction(img_path, checkpoint_path=checkpoint_path)
        filename = osp.basename(img_path)
        fpath = osp.join(saveDir, filename,)
        fig.save(fpath)
        if i > 50:
          break

def eval_on_coco():

    dirpath = '/home/zhenganlin/june/MSCOCO/'
    transform = stage1_transform(img_size=256, is_train=False, scale=0.8)
    dataset = CoCo(dirpath, 'val2017', transform=transform)
    dataloader = DataLoader(dataset, batch_size=25, shuffle=False, num_workers=2, pin_memory=False, drop_last=False)

    device = 'cuda'
    model_name = 'vit-m-vqgan'
    checkpoint_path = 'output/models/vit_vq_step_39000.pt'
    model = create_model(arch='vqgan', version=model_name, pretrained=True, checkpoint_path=checkpoint_path)
    model.eval()

    saveDir = 'reconstructions'
    os.makedirs(saveDir, exist_ok=True)

    model.to(device)
    titles = ['origin', 'reconstruct']
    for i, (images, _) in enumerate(dataloader):
        
        bs = images.size(0) * i
        images = images.to(device)
        with torch.no_grad():
            gen_imgs, _ = model(images)
        print('iter-{}, gen_imgs.shape:{}'.format(i, gen_imgs.shape))
        for k, (img, re) in enumerate(zip(images, gen_imgs)):
            
            img = restore(img)
            rec = restore(re)
            h, w = img.size

            fig = Image.new("RGB", (w, h))
            fig.paste(rec, (0,0))
            fpath = osp.join(saveDir, '{:06d}.png'.format(bs + k))
            fig.save(fpath)
    coco_eval()

def coco_eval(gtDir = None, saveDir = None):

    if (gtDir is None) | (saveDir is None):
        # Perform evaluation on the generated images.
        gtDir = '/home/zhenganlin/june/MSCOCO/coco-val'
        saveDir = 'reconstructions'

    gen_names = os.listdir(saveDir)
    img_names = os.listdir(gtDir)

    assert len(gen_names) == len(img_names), \
        f"generate only {len(gen_names)} images, while there are {len(img_names)} in total!"

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=saveDir,
        input2=gtDir,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=True,
    )
    fid = metrics_dict['frechet_inception_distance']
    inception_score = metrics_dict['inception_score_mean']
    print('FID:{:.4f}, IS: {:.4f}'.format(fid, inception_score))

if __name__ == '__main__':

    eval_on_coco()
