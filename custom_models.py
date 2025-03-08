#--------------------------------
# A collection of two classes from nn.Module to initialize convolutional encoders and decoders.
# The __init__ method has several parameter for fine control of the architecture
#--------------------------------

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import os 
from torchsummary import summary
import torchvision.transforms.v2 as transforms

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(),]), p=0.3)

dataaugm = transforms.Compose([
    transforms.RandomAffine(degrees=(-4.0, 4.0)),
    transforms.RandomResizedCrop(size = (128, 128) , scale = (0.93, 1.06)), 
    transforms.RandomAdjustSharpness(sharpness_factor = 2.0, p=0.5),
    transforms.RandomAdjustSharpness(sharpness_factor = 0.1, p=0.5),
    transforms.ColorJitter(brightness = 0.3 , contrast = 0.5, saturation = 0.9),
    transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianNoise(mean = 0.0, sigma = 0.12, clip=True),]), p=0.25),
    transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianNoise(mean = 0.0, sigma = 0.08, clip=True),]), p=0.25),
    transforms.RandomErasing(p = 0.5, scale = (0.02, 0.06), ratio = (0.5, 2.0), value = 0),
    # transforms.Resize(size = (128, 128) )
    ])

blurme = transforms.Compose([
    transforms.GaussianBlur(kernel_size = 7, sigma=(0.7, 0.9)),
    ])



class SpectroImageDataset(Dataset):

    def __init__(self, imgpath):
        self.all_img_files = [a for a in os.listdir(imgpath) if '.png' in a]
        self.imgpath = imgpath

    def __getitem__(self, index):     
        img = Image.open( os.path.join(self.imgpath,  self.all_img_files[index] ))

        x_orig = pil_to_tensor(img).to(torch.float32) / 255.0
        x_augm = dataaugm(x_orig)

        # simple denoising with threshold 
        # thld = x_orig.quantile(q=0.95)
        x_orig = blurme(x_orig)
        thld = 0.30
        x_orig[x_orig < thld] = 0.0

        y = self.all_img_files[index]
        return (x_orig, x_augm, y)
    
    def __len__(self):
        return (len(self.all_img_files))




class EncoderAvgpool(nn.Module):

    def __init__(self):
        super(EncoderAvgpool, self).__init__()
        n_ch_in = 1
        ch = [64, 128, 256, 512]
        po = [(2, 2), (4, 2), (4, 2), (4, 2)]
        self.padding =  "same"

        self.conv0 = nn.Sequential(
            nn.Conv2d(n_ch_in,  ch[0], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.Conv2d(ch[0], ch[0], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(),
            nn.AvgPool2d(po[0], stride=po[0])
            )
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.Conv2d(ch[1], ch[1], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(),
            nn.AvgPool2d(po[1], stride=po[1])
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch[1], ch[2], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.Conv2d(ch[2], ch[2], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(),
            nn.AvgPool2d(po[2], stride=po[2])
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch[2], ch[3], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.Conv2d(ch[3], ch[3], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.AvgPool2d(po[3], stride=po[3])
            )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return(x)








class EncoderNopad(nn.Module):

    def __init__(self):
        super(EncoderNopad, self).__init__()
        n_ch_in = 1
        ch = [64, 128, 256, 512]
        po = [(2, 2), (2, 2), (3, 2), (3, 1)]
    
        self.conv0 = nn.Sequential(
            nn.Conv2d(n_ch_in,  ch[0], kernel_size=(3,3), stride=1, padding=0),
            nn.Conv2d(ch[0], ch[0], kernel_size=(3,3), stride=1, padding=0),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(),
            nn.AvgPool2d(po[0], stride=po[0])
            )
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], kernel_size=(3,3), stride=1, padding=0),
            nn.Conv2d(ch[1], ch[1], kernel_size=(3,3), stride=1, padding=0),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(),
            nn.AvgPool2d(po[1], stride=po[1])
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch[1], ch[2], kernel_size=(3,3), stride=1, padding=0),
            nn.Conv2d(ch[2], ch[2], kernel_size=(3,3), stride=1, padding=0),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(),
            nn.AvgPool2d(po[2], stride=po[2])
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch[2], ch[3], kernel_size=(3,3), stride=1, padding=0),
            nn.Conv2d(ch[3], ch[3], kernel_size=(3,3), stride=1, padding=0),
            nn.AvgPool2d(po[3], stride=po[3])
            )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return(x)



class EncoderSimple(nn.Module):

    def __init__(self):
        super(EncoderSimple, self).__init__()
        n_ch_in = 1
        ch = [64, 128, 256, 512]
        po = [(2, 2), (4, 2), (4, 2), (4, 2)]

        self.conv0 = nn.Sequential(
            nn.Conv2d(n_ch_in, ch[0], 3, stride=po[0], bias=False, padding=1),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(True),)
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], 3, stride=po[1], bias=False, padding=1),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(True),)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch[1], ch[2], 3, stride=po[2], bias=False, padding=1),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(True),)
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch[2], ch[3], 3, stride=po[3], bias=False, padding=1),
            )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return(x)





class DecoderTransp(nn.Module):
    def __init__(self) :
        super(DecoderTransp, self).__init__()
        n_ch_out=3
        ch =  [512, 256, 128, 64]
        po =  [(2, 2), (4, 2), (4, 2), (4, 2)]
           
        self.tconv0 = nn.Sequential(
            nn.ConvTranspose2d(ch[0], ch[0], kernel_size=(3,3), stride=po[0], padding=(0,0), output_padding=(0,0)), 
            nn.BatchNorm2d(ch[0]),
            nn.ReLU()
            )
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(ch[0], ch[1], kernel_size=(5,5), stride=po[1], padding=(2,2), output_padding=(0,0)), 
            nn.BatchNorm2d(ch[1]),
            nn.ReLU()
            )
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(ch[1], ch[2], kernel_size=(5,5), stride=po[2], padding=(2,2), output_padding=(0,0)), 
            nn.BatchNorm2d(ch[2]),
            nn.ReLU()
            )
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(ch[2], ch[3], kernel_size=(5,5), stride=po[3], padding=(3,3),  output_padding=(1,1)), 
            nn.BatchNorm2d(ch[3]),
            nn.ReLU()
            )
        self.out_map = nn.Sequential(
            nn.Conv2d(ch[3], n_ch_out, kernel_size=(1,1), padding=0),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.tconv0(x)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.out_map(x)
        return x





class DecoderUpsample(nn.Module):

    def __init__(self):
        super(DecoderUpsample, self).__init__()
        n_ch_out=3
        ch =  [512, 256, 128, 64]
        po =  [(2, 2), (4, 2), (4, 2), (4, 2)]
    
        # 'bilinear', 'bicubic'
        self.tconv0 = nn.Sequential(
            nn.Upsample(scale_factor = po[0], mode='bilinear'),
            nn.Conv2d(ch[0], ch[0], kernel_size=(3,3), stride=1, padding='same'),
            # nn.Conv2d(ch[0], ch[0], kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU()
            )
        self.tconv1 = nn.Sequential(
            nn.Upsample(scale_factor = po[1], mode='bilinear'),
            nn.Conv2d(ch[0], ch[1], kernel_size=(3,3), stride=1, padding='same'),
            # nn.Conv2d(ch[1], ch[1], kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU()
            )
        self.tconv2 = nn.Sequential(
            nn.Upsample(scale_factor = po[2], mode='bilinear'),
            nn.Conv2d(ch[1], ch[2], kernel_size=(3,3), stride=1, padding='same'),
            # nn.Conv2d(ch[2], ch[2], kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU()
            )
        self.tconv3 = nn.Sequential(
            nn.Upsample(scale_factor = po[3], mode='bilinear'),
            nn.Conv2d(ch[2], ch[3], kernel_size=(3,3), stride=1, padding='same'),
            # nn.Conv2d(ch[3], ch[3], kernel_size=(3,3), stride=1, padding='same'),
            nn.BatchNorm2d(ch[3]),
            nn.ReLU()
            )
        self.out_map = nn.Sequential(
            nn.Conv2d(ch[3], n_ch_out, kernel_size=(1,1), padding=0),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.tconv0(x)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.out_map(x)
        return x


# devel code - supress execution if this is imported as module 
if __name__ == "__main__":

    model_enc = EncoderAvgpool() 
    model_enc = model_enc.to(device)
    summary(model_enc, (1, 128, 128))

    model_enc = EncoderNopad() 
    model_enc = model_enc.to(device)
    summary(model_enc, (1, 128, 128))

    model_enc = EncoderSimple()
    model_enc = model_enc.to(device)
    summary(model_enc, (1, 128, 128))



    model_dec = DecoderTransp()
    model_dec = model_dec.to(device)
    summary(model_dec, (512, 1, 8))

    model_dec = DecoderUpsample()
    model_dec = model_dec.to(device)
    summary(model_dec, (512, 1, 8))

  






