#--------------------------------
#
# 
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



par = {
    'da': {
        'rot_deg' : 0.3,
        'trans_prop' : 0.01,
        'brightness' : 0.3,

        },


    }

# par['da']['rot_deg'] = 15.0
# par['da']['trans_prop'] = 0.3
par['da']['brightness'] = 0.9



class SpectroImageDataset(Dataset):

    def __init__(self, imgpath, augment_1=False, augment_2=False):
        self.all_img_files = [a for a in os.listdir(imgpath) if '.png' in a]
        self.imgpath = imgpath
        self.augment_1 = augment_1
        self.augment_2 = augment_2


        self.dataaugm = transforms.Compose([
            transforms.RandomAffine(translate=(par['da']['trans_prop'], 0.0), degrees=(-par['da']['rot_deg'], par['da']['rot_deg'])),
            transforms.RandomAdjustSharpness(sharpness_factor = 2.0, p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor = 0.1, p=0.5),
            transforms.ColorJitter(brightness = par['da']['brightness'] , contrast = 0.5, saturation = 0.9),
            transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianNoise(mean = 0.0, sigma = 0.10, clip=True),]), p=0.20),
            transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianNoise(mean = 0.0, sigma = 0.05, clip=True),]), p=0.25),
            ])

        self.blurme = transforms.Compose([
            transforms.GaussianBlur(kernel_size = 7, sigma=(0.7, 0.9)),
            ])
       
    def __getitem__(self, index):     
        img = Image.open( os.path.join(self.imgpath,  self.all_img_files[index] ))

        x_1 = pil_to_tensor(img).to(torch.float32) / 255.0

        if self.augment_2:
            x_2 = self.dataaugm(x_1)
        else:    
            x_2 = x_1

        # simple de-noising with threshold 
        x_1 = self.blurme(x_1)
        thld = 0.30
        x_1[x_1 < thld] = 0.0

        if self.augment_1: 
            x_1 = self.dataaugm(x_1)   

        # prepare meta.data too
        y = self.all_img_files[index]

        return (x_1, x_2, y)
    
    def __len__(self):
        return (len(self.all_img_files))




