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
from plotly.subplots import make_subplots
import plotly.express as px

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SpectroImageDataset(Dataset):

    def __init__(self, imgpath, par = None, augment_1=False, augment_2=False, denoise_1=False, denoise_2=False):
        self.all_img_files = [a for a in os.listdir(imgpath) if '.png' in a]
        self.imgpath = imgpath
        self.par = par
        self.augment_1 = augment_1
        self.augment_2 = augment_2
        self.denoise_1 = denoise_1
        self.denoise_2 = denoise_2

        if self.augment_1 or self.augment_2 or self.denoise_1 or self.denoise_2:
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








def make_data_augment_examples(pt_dataset, batch_size = 12):
    """
    # assess one realization of data augmentation 
    pt_dataset : an instance of torch.utils.data.Dataset
    """
    pt_loader = torch.utils.data.DataLoader(pt_dataset, batch_size=batch_size,  shuffle=False, drop_last=True)
    # take only first batch 
    for i, (da_1, da_2, fi) in enumerate(pt_loader, 0):
        if i > 0: break
    fig = make_subplots(rows=batch_size, cols=2)
    for ii in range(batch_size): 
        # print(ii)
        img_1 = da_1[ii].cpu().detach().numpy()
        img_1 = img_1.squeeze() 
        img_1 = 255*img_1 
        img_2 = da_2[ii].cpu().detach().numpy()
        img_2 = img_2.squeeze() 
        img_2 = 255*img_2 
        fig.add_trace(px.imshow(img_1).data[0], row=ii+1, col=1)
        fig.add_trace(px.imshow(img_2).data[0], row=ii+1, col=2)
    fig.update_layout(autosize=True,height=400*batch_size, width = 2000)
    return(fig)







