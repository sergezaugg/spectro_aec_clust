#--------------------------------
#
# 
#--------------------------------

import torch
import numpy as np
# import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import os 
# from torchsummary import summary
import torchvision.transforms.v2 as transforms

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataaugm = transforms.Compose([
    # transforms.RandomAffine(degrees=(-4.0, 4.0)),
    # transforms.RandomResizedCrop(size = (128, 128) , scale = (0.93, 1.06)), 
    transforms.RandomAdjustSharpness(sharpness_factor = 2.0, p=0.5),
    transforms.RandomAdjustSharpness(sharpness_factor = 0.1, p=0.5),
    transforms.ColorJitter(brightness = 0.3 , contrast = 0.5, saturation = 0.9),
    transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianNoise(mean = 0.0, sigma = 0.10, clip=True),]), p=0.20),
    transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianNoise(mean = 0.0, sigma = 0.05, clip=True),]), p=0.25),
    # transforms.RandomErasing(p = 0.5, scale = (0.02, 0.06), ratio = (0.5, 2.0), value = 0),
    # transforms.Resize(size = (128, 128) )
    ])

blurme = transforms.Compose([
    transforms.GaussianBlur(kernel_size = 7, sigma=(0.7, 0.9)),
    ])

class SpectroImageDataset(Dataset):

    def __init__(self, imgpath, edge_attenuation = True):
        self.all_img_files = [a for a in os.listdir(imgpath) if '.png' in a]
        self.imgpath = imgpath
        self.edge_attenuation = edge_attenuation
        # make attenuation window (hanning-like)
        edge_att_win = np.kaiser(128, 5)
        edge_att_win = edge_att_win - edge_att_win.min()
        edge_att_win = edge_att_win / edge_att_win.max()
        edge_att_win = np.broadcast_to(edge_att_win, shape = (128,128))
        att = torch.from_numpy(edge_att_win)
        self.att = att.type(torch.float32)
  
    def __getitem__(self, index):     
        img = Image.open( os.path.join(self.imgpath,  self.all_img_files[index] ))

        x_orig = pil_to_tensor(img).to(torch.float32) / 255.0
        x_augm = dataaugm(x_orig)

        # simple denoising with threshold 
        # thld = x_orig.quantile(q=0.95)
        x_orig = blurme(x_orig)
        thld = 0.30
        x_orig[x_orig < thld] = 0.0

        # # apply attenuation window
        if self.edge_attenuation:
            x_orig = x_orig*self.att
            x_augm = x_augm*self.att

        # prepare meta.data too
        y = self.all_img_files[index]

        return (x_orig, x_augm, y)
    
    def __len__(self):
        return (len(self.all_img_files))




