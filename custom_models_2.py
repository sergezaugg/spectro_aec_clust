#--------------------------------
# A collection of two classes from nn.Module to initialize convolutional encoders and decoders.
# The __init__ method has several parameter for fine control of the architecture
#--------------------------------

import torch
# import numpy as np
import torch.nn as nn
# from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
# from PIL import Image
# import os 
from torchsummary import summary
# import torchvision.transforms.v2 as transforms

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class EncoderAvgpool(nn.Module):
    def __init__(self):
        super(EncoderAvgpool, self).__init__()
        n_ch_in = 1
        ch = [64, 128, 256, 256, 128]
        po = [(2, 2), (4, 2), (4, 2), (2, 2), (2, 2)]
        self.padding =  "same"
        self.conv0 = nn.Sequential(
            nn.Conv2d(n_ch_in,  ch[0], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.Conv2d(ch[0], ch[0], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(),
            nn.AvgPool2d(po[0], stride=po[0]))
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.Conv2d(ch[1], ch[1], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(),
            nn.AvgPool2d(po[1], stride=po[1]))
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch[1], ch[2], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.Conv2d(ch[2], ch[2], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(),
            nn.AvgPool2d(po[2], stride=po[2]))
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch[2], ch[3], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.Conv2d(ch[3], ch[3], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[3]),
            nn.ReLU(),
            nn.AvgPool2d(po[3], stride=po[3]))
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch[3], ch[4], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.Conv2d(ch[4], ch[4], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.AvgPool2d(po[4], stride=po[4]))
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return(x)
    


class DecoderTranspNew(nn.Module):
    def __init__(self) :
        super(DecoderTranspNew, self).__init__()
        n_ch_out=1
        ch =  [128, 256, 128, 128, 64]
        # po =  [(4, 2), (4, 2), (2, 2), (2, 2), (2, 2)]
        po =  [(2, 2), (2, 2), (4, 2), (4, 2), (2, 2)]
           
        self.tconv0 = nn.Sequential(
            # nn.ConvTranspose2d(ch[0], ch[0], kernel_size=(5,5), stride=po[0], padding=(1,2), output_padding=(1,1)), 
            nn.ConvTranspose2d(ch[0], ch[0], kernel_size=(5,5), stride=po[0], padding=(2,2), output_padding=(1,1)), 

            nn.BatchNorm2d(ch[0]),
            nn.ReLU()
            )
        self.tconv1 = nn.Sequential(
            # nn.ConvTranspose2d(ch[0], ch[1], kernel_size=(5,5), stride=po[1], padding=(1,2), output_padding=(1,1)),
            nn.ConvTranspose2d(ch[0], ch[1], kernel_size=(5,5), stride=po[1], padding=(2,2), output_padding=(1,1)), 
            nn.BatchNorm2d(ch[1]),
            nn.ReLU()
            )
        self.tconv2 = nn.Sequential(
            # nn.ConvTranspose2d(ch[1], ch[2], kernel_size=(5,5), stride=po[2], padding=(2,2), output_padding=(1,1)),
            nn.ConvTranspose2d(ch[1], ch[2], kernel_size=(5,5), stride=po[2], padding=(1,2), output_padding=(1,1)),  
            nn.BatchNorm2d(ch[2]),
            nn.ReLU()
            )
        self.tconv3 = nn.Sequential(
            # nn.ConvTranspose2d(ch[2], ch[3], kernel_size=(5,5), stride=po[3], padding=(2,2),  output_padding=(1,1)), 
            nn.ConvTranspose2d(ch[2], ch[3], kernel_size=(5,5), stride=po[3], padding=(1,2),  output_padding=(1,1)), 
            nn.BatchNorm2d(ch[3]),
            nn.ReLU()
            )
        self.tconv4 = nn.Sequential(
            # nn.ConvTranspose2d(ch[3], ch[4], kernel_size=(5,5), stride=po[4], padding=(2,2),  output_padding=(1,1)), 
            nn.ConvTranspose2d(ch[3], ch[4], kernel_size=(5,5), stride=po[4], padding=(2,2),  output_padding=(1,1)), 
            nn.BatchNorm2d(ch[4]),
            nn.ReLU()
            )
        self.out_map = nn.Sequential(
            nn.Conv2d(ch[4], n_ch_out, kernel_size=(1,1), padding=0),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.tconv0(x)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.out_map(x)
        return x




# devel code - supress execution if this is imported as module 
if __name__ == "__main__":

    model_enc = EncoderAvgpool() 
    model_enc = model_enc.to(device)
    # summary(model_enc, (1, 128, 256))
    summary(model_enc, (1, 128, 1152))

    model_dec = DecoderTranspNew()
    model_dec = model_dec.to(device)
    summary(model_dec, (128, 1, 36))





    # get size of receptive field 

    class Convnet00(nn.Module):
        def __init__(self):
            super(Convnet00, self).__init__()
            ch = 55
            po = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
            self.conv0 = nn.Sequential(
                nn.Conv2d(1,  ch, kernel_size=(5,5), stride=1, padding=0),
                nn.AvgPool2d(po[0], stride=po[0]))
            self.conv1 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=(5,5), stride=1, padding=0),
                nn.AvgPool2d(po[1], stride=po[1]))
            self.conv2 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=(5,5), stride=1, padding=0),
                nn.AvgPool2d(po[2], stride=po[2]))
            self.conv3 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=(5,5), stride=1, padding=0),
                nn.AvgPool2d(po[3], stride=po[3]))
            self.conv4 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=(5,5), stride=1, padding=0),
                nn.AvgPool2d(po[4], stride=po[4]))
            self.conv5 = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=(5,5), stride=1, padding=0),
                nn.AvgPool2d(po[5], stride=po[5]))
        def forward(self, x):
            x = self.conv0(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            return(x)
        
    model_conv = Convnet00() 
    model_conv = model_conv.to(device)
    summary(model_conv, (1, 316, 316))






