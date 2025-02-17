#----------------------
# Author : Serge Zaugg
# Description : misc pytorch classes to be imported in main script
#----------------------

import torch
from torch.utils.data import Dataset
import os 
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import os 
import torchvision.transforms as transforms
from torchsummary import summary


dataaugm = transforms.Compose([
   transforms.RandomAffine(degrees=(-1.0, 1.0), translate=(0.02,0.02), scale=(0.97,1.03)),
    #    transforms.RandomCrop(size=120, padding=None, pad_if_needed=False)
    ])

# Create a dataset for png images in a folder
class SpectroImageDataset(Dataset):
    def __init__(self, imgpath):
        self.all_img_files = [a for a in os.listdir(imgpath) if '.png' in a]
        self.imgpath = imgpath
    def __getitem__(self, index):     
        img = Image.open( os.path.join(self.imgpath,  self.all_img_files[index] ))
        # img = dataaugm(img)
        img = img.resize((128, 128))
        x = pil_to_tensor(img).to(torch.float32) / 255.0
        y = self.all_img_files[index]
        return (x ,y)
    def __len__(self):
        return (len(self.all_img_files))




torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):

    def __init__(self, 
                 n_ch_in = 3, 
                 padding = "same",
                 ch = [16, 32, 64, 128, 256], 
                 co = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
                 po = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
                 n_ch_latent = 1024, 
                 flattened_size = 256,
                 incl_last_layer = True,
                 ):
        """
        """
        
        super(Encoder, self).__init__()

        self.incl_last_layer = incl_last_layer

        # conv block 0
        self.conv0 = nn.Sequential(
            nn.Conv2d(n_ch_in,  ch[0], kernel_size=co[0], stride=1, padding=padding),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(),
            nn.AvgPool2d(po[0], stride=po[0])
            )
       
        # conv block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], kernel_size=co[1], stride=1, padding=padding),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(),
            nn.AvgPool2d(po[1], stride=po[1])
            )
       
        # conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch[1], ch[2], kernel_size=co[2], stride=1, padding=padding),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(),
            nn.AvgPool2d(po[2], stride=po[2])
            )
    
        # conv block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch[2], ch[3], kernel_size=co[3], stride=1, padding=padding),
            nn.BatchNorm2d(ch[3]),
            nn.ReLU(),
            nn.AvgPool2d(po[3], stride=po[3])
            )
     
        # conv block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch[3], ch[4], kernel_size=co[4], stride=1, padding=padding),
            nn.BatchNorm2d(ch[4]),
            nn.ReLU(),
            nn.AvgPool2d(po[4], stride=po[4])
            )
    

        self.flatn = nn.Flatten()

        self.fc0 = nn.Sequential(
            nn.Linear(flattened_size, n_ch_latent),
            nn.ReLU(),
            )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatn(x)
        print('flattened_size_2', x.size())
        if self.incl_last_layer:
            x = self.fc0(x)
        return(x)










class Decoder(nn.Module):
    def __init__(self, n_ch_out=3, n_ch_latent=256, shape_output = (32, 32), n_conv_blocks=4,
                 ch = [256, 128, 64, 32, 16], po = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]):
        
        super().__init__()

        self.n_conv_blocks = n_conv_blocks
        self.sho = shape_output
        self.chout = n_ch_out

        if self.n_conv_blocks == 5:
            aa = self.sho[0] // (po[0][0]*po[1][0]*po[2][0]*po[3][0]*po[4][0])
            bb = self.sho[1] // (po[0][1]*po[1][1]*po[2][1]*po[3][1]*po[4][1])
            last_ch_depth = ch[0]
        if self.n_conv_blocks == 4:
            aa = self.sho[0] // (po[0][0]*po[1][0]*po[2][0]*po[3][0])
            bb = self.sho[1] // (po[0][1]*po[1][1]*po[2][1]*po[3][1])
            last_ch_depth = ch[0]
        if self.n_conv_blocks == 3:
            aa = self.sho[0] // (po[1][0]*po[2][0]*po[3][0])
            bb = self.sho[1] // (po[1][1]*po[2][1]*po[3][1])
            last_ch_depth = ch[1]
        if self.n_conv_blocks == 2:
            aa = self.sho[0] // (po[2][0]*po[3][0])
            bb = self.sho[1] // (po[2][1]*po[3][1])
            last_ch_depth = ch[2]
        if self.n_conv_blocks == 1:
            aa = self.sho[0] // (po[3][0])
            bb = self.sho[1] // (po[3][1])
            last_ch_depth = ch[3]
        if self.n_conv_blocks == 0:
            aa = self.sho[0] 
            bb = self.sho[1]
            last_ch_depth = self.chout
            
        flattened_size = aa * bb * last_ch_depth # ch[0]
        print('flattened_size', flattened_size)
        # rewrap array into a shape for conv
        cd1 = flattened_size // (aa*last_ch_depth)
        cd2 = flattened_size // (bb*last_ch_depth)
       
        self.linear = nn.Sequential(
            nn.Linear(n_ch_latent, flattened_size),
            nn.ReLU(),
            )
        
        print('reshape to: ', last_ch_depth, cd1, cd2)
        self.unfla = nn.Unflatten(1, (last_ch_depth, cd1, cd2))

        # transpose conv block 0
        if self.n_conv_blocks >= 5:
            self.tconv0 = nn.Sequential(
                nn.ConvTranspose2d(ch[0], ch[0], kernel_size=(3,3), padding=1),
                nn.ConvTranspose2d(ch[0], ch[0], kernel_size=(3,3), padding=1, stride=po[0], output_padding=1), 
                nn.BatchNorm2d(ch[0]),
                nn.ReLU()
                )
        else:
            self.tconv0 = nn.Identity()

        # transpose conv block 1
        if self.n_conv_blocks >= 4:
            self.tconv1 = nn.Sequential(
                nn.ConvTranspose2d(ch[0], ch[1], kernel_size=(3,3), padding=1),
                nn.ConvTranspose2d(ch[1], ch[1], kernel_size=(3,3), padding=1, stride=po[1], output_padding=1), 
                nn.BatchNorm2d(ch[1]),
                nn.ReLU()
                )
        else:
            self.tconv1 = nn.Identity()
        
        # transpose conv block 2
        if self.n_conv_blocks >= 3:
            self.tconv2 = nn.Sequential(
                nn.ConvTranspose2d(ch[1], ch[2], kernel_size=(3,3), padding=1),
                nn.ConvTranspose2d(ch[2], ch[2], kernel_size=(3,3), padding=1, stride=po[2], output_padding=1), 
                nn.BatchNorm2d(ch[2]),
                nn.ReLU()
                )
        else:
            self.tconv2 = nn.Identity()

        # transpose conv block 3
        if self.n_conv_blocks >= 2:
            self.tconv3 = nn.Sequential(
                nn.ConvTranspose2d(ch[2], ch[3], kernel_size=(3,3), padding=1),
                nn.ConvTranspose2d(ch[3], ch[3], kernel_size=(3,3), padding=1, stride=po[3], output_padding=1), 
                nn.BatchNorm2d(ch[3]),
                nn.ReLU()
                )
        else:
            self.tconv3 = nn.Identity()
        
       # transpose conv block 4
        if self.n_conv_blocks >= 1:
            self.tconv4 = nn.Sequential(
                nn.ConvTranspose2d(ch[3], ch[4], kernel_size=(3,3), padding=1),
                nn.ConvTranspose2d(ch[4], ch[4], kernel_size=(3,3), padding=1, stride=po[4], output_padding=1), 
                nn.BatchNorm2d(ch[4]),
                nn.ReLU()
                )
        else:
            self.tconv4 = nn.Identity()

        if self.n_conv_blocks >= 1:
            self.out_map = nn.Sequential(
                nn.Conv2d(ch[4], self.chout, kernel_size=(1,1), padding=0),
                nn.Sigmoid()
                )
        else:
            self.out_map = nn.Sequential(
                nn.Conv2d(self.chout, self.chout, kernel_size=(1,1), padding=0),
                nn.Sigmoid()
                )

        # self.relu = nn.ReLU()
     
    def forward(self, x):
        x = self.linear(x)
        x = self.unfla(x)
        x = self.tconv0(x)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.out_map(x)
        return x




# devel code - supress execution if this is imported as module 
if __name__ == "__main__":

    model_enc = Encoder(n_ch_in = 3, 
                        padding = "same",
                        ch = [32, 64, 128, 256, 64],
                        co = [(5, 5), (5, 5), (3, 3), (3, 3), (3, 3)],
                        po = [(2, 2), (2, 1), (2, 1), (2, 1), (1, 1)],
                        n_ch_latent = 1024, 
                        flattened_size = 2048,
                        incl_last_layer = True,
                        ) 
    
    model_enc = Encoder() 
    model_enc = model_enc.to(device)
    summary(model_enc, (3, 32, 32))







    impsha = (32, 32)
    latsha = 1024


    model_dec = Decoder(n_ch_out=3, 
                        n_ch_latent=latsha, 
                        shape_output = impsha, 
                        n_conv_blocks = 5,
                        ch = [128, 96, 64, 32, 16],
                        po = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
                        )
    model_dec = model_dec.to(device)
    summary(model_dec, (1024,))



