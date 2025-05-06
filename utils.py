#--------------------------------
# Author : Serge Zaugg
# Description : 
#--------------------------------

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import os 
import torchvision.transforms.v2 as transforms

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SpectroImageDataset(Dataset):

    def __init__(self, imgpath, par, augment_1=False, augment_2=False, denoise_1=False, denoise_2=False):
        self.all_img_files = [a for a in os.listdir(imgpath) if '.png' in a]
        self.imgpath = imgpath
        self.par = par
        self.augment_1 = augment_1
        self.augment_2 = augment_2
        self.denoise_1 = denoise_1
        self.denoise_2 = denoise_2

        self.dataaugm = transforms.Compose([
            transforms.RandomAffine(translate=(self.par['da']['trans_prop'], 0.0), degrees=(-self.par['da']['rot_deg'], self.par['da']['rot_deg'])),
            transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianNoise(mean = 0.0, sigma = self.par['da']['gnoisesigm'], clip=True),]), p=self.par['da']['gnoiseprob']),
            transforms.ColorJitter(brightness = self.par['da']['brightness'] , contrast = self.par['da']['contrast']),
            ])
 
    def __getitem__(self, index):     
        img = Image.open( os.path.join(self.imgpath,  self.all_img_files[index] ))
        # load pimage and set range to [0.0, 1.0]
        x_1 = pil_to_tensor(img).to(torch.float32) / 255.0
        x_2 = pil_to_tensor(img).to(torch.float32) / 255.0
        # simple de-noising with threshold
        if self.denoise_1: 
            x_1[x_1 < self.par['den']['thld'] ] = 0.0
        if self.denoise_2: 
            x_2[x_2 < self.par['den']['thld'] ] = 0.0    
        # data augmentation 
        if self.augment_1: 
            x_1 = self.dataaugm(x_1)  
        if self.augment_2:
            x_2 = self.dataaugm(x_2) 
        # prepare meta-data 
        y = self.all_img_files[index]

        return (x_1, x_2, y)
    
    def __len__(self):
        return (len(self.all_img_files))





