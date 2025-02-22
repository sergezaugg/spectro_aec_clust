#--------------------------------
# A collection of two classes from nn.Module to initialize convolutional encoders and decoders.
# The __init__ method has several parameter for fine control of the architecture
#--------------------------------

import numpy as np
import torch
import numpy as np
from torchsummary import summary
import torch.nn as nn
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

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



dataaugm = transforms.Compose([
    transforms.RandomAffine(degrees=(-1.0, 1.0), translate=(0.02,0.02), scale=(0.98,1.02)),
    # transforms.RandomCrop(size=126, padding=None, pad_if_needed=False)
    ])

# Create a dataset for png images in a folder
class SpectroImageDataset(Dataset):
    def __init__(self, imgpath):
        self.all_img_files = [a for a in os.listdir(imgpath) if '.png' in a]
        self.imgpath = imgpath
    def __getitem__(self, index):     
        img = Image.open( os.path.join(self.imgpath,  self.all_img_files[index] ))
        # img = dataaugm(img)
        # img = img.resize((128, 128))
        x = pil_to_tensor(img).to(torch.float32) / 255.0
        y = self.all_img_files[index]
        return (x ,y)
    def __len__(self):
        return (len(self.all_img_files))





class Encoder(nn.Module):

    def __init__(self, n_ch_in, n_ch_latent, shape_input, n_conv_blocks, padding = "same",
                 ch = [16, 32, 64, 128, 256], po = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]):
        """
        Initialize a convolutional encoder (first half of an auto-encoder)
        Parameters:
        n_ch_in (int): nb channels in input image
        n_ch_latent (int): size of the latent vector
        shape_input (list): width an height of the input image, e.g. [32,32]
        n_conv_blocks (int): number of convolutional blocks, should be in 0 to 5
        padding (str): set to "same"
        ch (list): Nb of channels in each convolutional block, list of length n_conv_blocks
        po (list): Pooling applied after each convolutional block, list of length n_conv_blocks
        """
        
        super(Encoder, self).__init__()

        self.shi = shape_input
        self.padding = padding
        self.n_conv_blocks = n_conv_blocks

        # conv block 0
        if self.n_conv_blocks >= 1:
            self.conv0 = nn.Sequential(
                nn.Conv2d(n_ch_in,  ch[0], kernel_size=(3,3), stride=1, padding=self.padding),
                nn.Conv2d(ch[0], ch[0], kernel_size=(3,3), stride=1, padding=self.padding),
                nn.BatchNorm2d(ch[0]),
                nn.ReLU(),
                nn.AvgPool2d(po[0], stride=po[0])
                )
        else:
            self.conv0 = nn.Identity()

        # conv block 1
        if self.n_conv_blocks >= 2:
            self.conv1 = nn.Sequential(
                nn.Conv2d(ch[0], ch[1], kernel_size=(3,3), stride=1, padding=self.padding),
                nn.Conv2d(ch[1], ch[1], kernel_size=(3,3), stride=1, padding=self.padding),
                nn.BatchNorm2d(ch[1]),
                nn.ReLU(),
                nn.AvgPool2d(po[1], stride=po[1])
                )
        else:
            self.conv1 = nn.Identity()
        
        # conv block 2
        if self.n_conv_blocks >= 3:
            self.conv2 = nn.Sequential(
                nn.Conv2d(ch[1], ch[2], kernel_size=(3,3), stride=1, padding=self.padding),
                nn.Conv2d(ch[2], ch[2], kernel_size=(3,3), stride=1, padding=self.padding),
                nn.BatchNorm2d(ch[2]),
                nn.ReLU(),
                nn.AvgPool2d(po[2], stride=po[2])
                )
        else:
            self.conv2 = nn.Identity()

        # conv block 3
        if self.n_conv_blocks >= 4:
            self.conv3 = nn.Sequential(
                nn.Conv2d(ch[2], ch[3], kernel_size=(3,3), stride=1, padding=self.padding),
                nn.Conv2d(ch[3], ch[3], kernel_size=(3,3), stride=1, padding=self.padding),
                nn.BatchNorm2d(ch[3]),
                nn.ReLU(),
                nn.AvgPool2d(po[3], stride=po[3])
                )
        else:
            self.conv3 = nn.Identity()    

        # conv block 4
        if self.n_conv_blocks >= 5:
            self.conv4 = nn.Sequential(
                nn.Conv2d(ch[3], ch[4], kernel_size=(3,3), stride=1, padding=self.padding),
                nn.Conv2d(ch[4], ch[4], kernel_size=(3,3), stride=1, padding=self.padding),
                nn.BatchNorm2d(ch[4]),
                nn.ReLU(),
                nn.AvgPool2d(po[4], stride=po[4])
                )
        else:
            self.conv4 = nn.Identity()    

        self.flatn = nn.Flatten()

        # flattened_size = int(self.ch1 * (self.shi[0]/(self.pool0[0]*self.pool1[0])) * (self.shi[1]/(self.pool0[1]*self.pool1[1])))
        if self.n_conv_blocks == 0:
            flattened_size = int(n_ch_in * (self.shi[0]) * (self.shi[1]))
        if self.n_conv_blocks == 1:
            flattened_size = int(ch[0] * (self.shi[0]/(po[0][0])) * (self.shi[1]/(po[0][1])))
        if self.n_conv_blocks == 2:
            flattened_size = int(ch[1] * (self.shi[0]/(po[0][0]*po[1][0])) * (self.shi[1]/(po[0][1]*po[1][1])))
        if self.n_conv_blocks == 3:
            flattened_size = int(ch[2] * (self.shi[0]/(po[0][0]*po[1][0]*po[2][0])) * (self.shi[1]/(po[0][1]*po[1][1]*po[2][1]))) 
        if self.n_conv_blocks == 4:
            flattened_size = int(ch[3] * (self.shi[0]/(po[0][0]*po[1][0]*po[2][0]*po[3][0])) * (self.shi[1]/(po[0][1]*po[1][1]*po[2][1]*po[3][1]))) 
        if self.n_conv_blocks == 5:
            flattened_size = int(ch[4] * (self.shi[0]/(po[0][0]*po[1][0]*po[2][0]*po[3][0]*po[4][0])) * (self.shi[1]/(po[0][1]*po[1][1]*po[2][1]*po[3][1]*po[4][1]))) 
        print('flattened_size_compute', flattened_size)

        self.dropout = nn.Dropout(0.5)

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
        self.dropout(x)
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
        
        print('reshape to: ', last_ch_depth, cd2, cd1)
        self.unfla = nn.Unflatten(1, (last_ch_depth, cd2, cd1))

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

    impsha = (128, 64)
    latsha = 512
    n_blck = 3

    model_enc = Encoder(n_ch_in = 1, 
                        n_ch_latent=latsha, 
                        shape_input = impsha, 
                        n_conv_blocks = n_blck,
                        ch = [16, 32, 64, 256, 512],
                        po = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
                        ) 
    model_enc = model_enc.to(device)
    summary(model_enc, (1, 128, 64))

    model_dec = Decoder(n_ch_out = 1, 
                        n_ch_latent=latsha, 
                        shape_output = impsha, 
                        n_conv_blocks = n_blck,
                        ch = [128, 64, 64, 32, 32],
                        po = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
                        )
    model_dec = model_dec.to(device)
    summary(model_dec, (latsha,))








