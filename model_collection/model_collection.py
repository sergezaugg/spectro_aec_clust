#--------------------------------
# A collection of two classes from nn.Module to initialize convolutional encoders and decoders.
# The __init__ method has several parameter for fine control of the architecture
#--------------------------------

import torch
import torch.nn as nn



# -------------------------------------------------
# GEN B0 (freq-pool : 128 - time-pool : 32)

class EncoderGenBTP32(nn.Module):
    def __init__(self, n_ch_in = 3, ch = [64, 128, 128, 128, 256]):
        super().__init__()
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
    
class DecoderGenBTP32(nn.Module):
    def __init__(self, n_ch_out=3, ch = [256, 128, 128, 128, 64]):
        super().__init__()
        po =  [(2, 2), (2, 2), (4, 2), (4, 2), (2, 2)]
        self.tconv0 = nn.Sequential(
            nn.ConvTranspose2d(ch[0], ch[0], kernel_size=(5,5), stride=po[0], padding=(2,2), output_padding=(1,1)), 
            nn.BatchNorm2d(ch[0]),
            nn.ReLU()
            )
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(ch[0], ch[1], kernel_size=(5,5), stride=po[1], padding=(2,2), output_padding=(1,1)), 
            nn.BatchNorm2d(ch[1]),
            nn.ReLU()
            )
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(ch[1], ch[2], kernel_size=(5,5), stride=po[2], padding=(1,2), output_padding=(1,1)),  
            nn.BatchNorm2d(ch[2]),
            nn.ReLU()
            )
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(ch[2], ch[3], kernel_size=(5,5), stride=po[3], padding=(1,2),  output_padding=(1,1)), 
            nn.BatchNorm2d(ch[3]),
            nn.ReLU()
            )
        self.tconv4 = nn.Sequential(
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

# -------------------------------------------------




# -------------------------------------------------
# GEN C (freq-pool : 128 - time-pool : 32)

class EncoderGenCTP32(nn.Module):
    def __init__(self, n_ch_in = 3, n_ch_out = 256, ch = [64, 128, 128, 256]):
        super().__init__()
        po = [(2, 2), (4, 2), (4, 2), (2, 2), (2, 2)]
        self.padding =  "same"
        conv_kernel = (3,3)
        self.conv0 = nn.Sequential(
            nn.Conv2d(n_ch_in,  ch[0], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.Conv2d(ch[0],  ch[0], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(),
            nn.AvgPool2d(po[0], stride=po[0]))
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.Conv2d(ch[1], ch[1], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(),
            nn.AvgPool2d(po[1], stride=po[1]))
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch[1], ch[2], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.Conv2d(ch[2], ch[2], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(),
            nn.AvgPool2d(po[2], stride=po[2]))
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch[2], ch[3], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.Conv2d(ch[3], ch[3], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[3]),
            nn.ReLU(),
            nn.AvgPool2d(po[3], stride=po[3]))
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch[3], n_ch_out, kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.Conv2d(n_ch_out, n_ch_out, kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(n_ch_out),
            nn.AvgPool2d(po[4], stride=po[4]))
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return(x)


class DecoderGenCTP32(nn.Module):
    def __init__(self, n_ch_in=256, n_ch_out=3, ch = [128, 128, 128, 64, 64]):
        super().__init__()
        self.padding =  "same"
        po =  [(2, 2), (2, 2), (4, 2), (4, 2), (2, 2)]
        self.tconv0 = nn.Sequential(
            nn.Upsample(scale_factor=po[0], mode='bilinear'),
            nn.Conv2d(n_ch_in,  n_ch_in, kernel_size=(3,3), stride=1, padding=self.padding),
            nn.Conv2d(n_ch_in,  ch[0], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(),
            )
        self.tconv1 = nn.Sequential(
            nn.Upsample(scale_factor=po[1], mode='bilinear'),
            nn.Conv2d(ch[0], ch[0], kernel_size=(3,3), stride=1, padding=self.padding), 
            nn.Conv2d(ch[0], ch[1], kernel_size=(3,3), stride=1, padding=self.padding), 
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(),
            )
        self.tconv2 = nn.Sequential(
            nn.Upsample(scale_factor=po[2], mode='bilinear'),
            nn.Conv2d(ch[1], ch[1], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.Conv2d(ch[1], ch[2], kernel_size=(3,3), stride=1, padding=self.padding),    
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(),
            )
        self.tconv3 = nn.Sequential(
            nn.Upsample(scale_factor=po[3], mode='bilinear'),
            nn.Conv2d(ch[2], ch[2], kernel_size=(3,3), stride=1, padding=self.padding), 
            nn.Conv2d(ch[2], ch[3], kernel_size=(3,3), stride=1, padding=self.padding), 
            nn.BatchNorm2d(ch[3]),
            nn.ReLU(),
            )
        self.out_map = nn.Sequential(
            nn.Upsample(scale_factor=po[4], mode='bilinear'),
            nn.Conv2d(ch[3], ch[3], kernel_size=(3,3), stride=1, padding=self.padding), 
            nn.Conv2d(ch[3], n_ch_out, kernel_size=(3,3), stride=1, padding=self.padding), 
            nn.BatchNorm2d(n_ch_out),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.tconv0(x)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.out_map(x)
        return x

# -------------------------------------------------









# -------------------------------------------------
# GEN B1 (freq-pool : 128 - time-pool : 16)

class EncoderGenBTP16(nn.Module):
    def __init__(self, n_ch_in = 3, ch = [64, 128, 128, 128, 256]):
        super(EncoderGenBTP16, self).__init__()
        po = [(2, 2), (4, 2), (4, 2), (2, 2), (2, 1)]
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
    
class DecoderGenBTP16(nn.Module):
    def __init__(self, n_ch_out = 3, ch =  [256, 128, 128, 128, 64]) :
        super(DecoderGenBTP16, self).__init__()
        po =  [(2, 2), (2, 2), (4, 2), (4, 2), (2, 1)]
        self.tconv0 = nn.Sequential(
            nn.ConvTranspose2d(ch[0], ch[0], kernel_size=(5,5), stride=po[0], padding=(2,2), output_padding=(1,1)), 
            nn.BatchNorm2d(ch[0]),
            nn.ReLU()
            )
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(ch[0], ch[1], kernel_size=(5,5), stride=po[1], padding=(2,2), output_padding=(1,1)), 
            nn.BatchNorm2d(ch[1]),
            nn.ReLU()
            )
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(ch[1], ch[2], kernel_size=(5,5), stride=po[2], padding=(1,2), output_padding=(1,1)),  
            nn.BatchNorm2d(ch[2]),
            nn.ReLU()
            )
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(ch[2], ch[3], kernel_size=(5,5), stride=po[3], padding=(2,2),  output_padding=(2,1)), 
            nn.BatchNorm2d(ch[3]),
            nn.ReLU()
            )
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(ch[3], ch[4], kernel_size=(5,5), stride=po[4], padding=(1,2),  output_padding=(1,0)), 
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

# -------------------------------------------------


# -------------------------------------------------
# GEN B2 (freq-pool : 128 - time-pool : 8)

class EncoderGenBTP08(nn.Module):
    def __init__(self, n_ch_in = 3, ch = [64, 128, 128, 128, 256]):
        super(EncoderGenBTP08, self).__init__()
        po = [(2, 2), (4, 2), (4, 2), (2, 1), (2, 1)]
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
    
class DecoderGenBTP08(nn.Module):
    def __init__(self, n_ch_out=3, ch = [256, 128, 128, 128, 64]) :
        super(DecoderGenBTP08, self).__init__()
        po =  [(2, 2), (2, 2), (4, 2), (4, 1), (2, 1)]
           
        self.tconv0 = nn.Sequential(
            nn.ConvTranspose2d(ch[0], ch[0], kernel_size=(5,5), stride=po[0], padding=(2,2), output_padding=(1,1)), 
            nn.BatchNorm2d(ch[0]),
            nn.ReLU()
            )
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(ch[0], ch[1], kernel_size=(5,5), stride=po[1], padding=(2,2), output_padding=(1,1)), 
            nn.BatchNorm2d(ch[1]),
            nn.ReLU()
            )
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(ch[1], ch[2], kernel_size=(5,5), stride=po[2], padding=(1,2), output_padding=(1,1)),  
            nn.BatchNorm2d(ch[2]),
            nn.ReLU()
            )
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(ch[2], ch[3], kernel_size=(5,5), stride=po[3], padding=(1,2),  output_padding=(0,0)), 
            nn.BatchNorm2d(ch[3]),
            nn.ReLU()
            )
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(ch[3], ch[4], kernel_size=(5,5), stride=po[4], padding=(1,2),  output_padding=(1,0)), 
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

# -------------------------------------------------


# -------------------------------------------------
# GEN B 3blocks - better for hi freq features - textures ? (freq-pool : 128 - time-pool : 8)

class EncoderGenB3blocks(nn.Module):
    def __init__(self, n_ch_in = 3, ch = [64, 128, 128, 256]):
        super().__init__()
        po = [(2, 2), (2, 2), (2, 2)]
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
            nn.Conv2d(ch[2], ch[3], kernel_size=(16,1), stride=1, padding='valid'),
            )   
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return(x)
    
class DecoderGenB3blocks(nn.Module):
    def __init__(self, n_ch_out=3, ch = [256, 128, 128, 128]):
        super().__init__()
        po =  [(2, 2), (2, 2), (2, 2)]
        self.tconv0 = nn.Sequential(
            nn.ConvTranspose2d(ch[0], ch[1], kernel_size=(16,1), stride=(1,1), padding=(0,0), output_padding=(0,0)), 
            nn.BatchNorm2d(ch[1]),
            nn.ReLU()
            )
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(ch[1], ch[2], kernel_size=(5,5), stride=po[1], padding=(2,2), output_padding=(1,1)), 
            nn.BatchNorm2d(ch[2]),
            nn.ReLU()
            )
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(ch[2], ch[3], kernel_size=(5,5), stride=po[2], padding=(2,2), output_padding=(1,1)),  
            nn.BatchNorm2d(ch[3]),
            nn.ReLU()
            )
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(ch[3], n_ch_out, kernel_size=(5,5), stride=po[2], padding=(2,2), output_padding=(1,1)),  
            nn.BatchNorm2d(n_ch_out),
            nn.Sigmoid()
            )
    def forward(self, x):
        x = self.tconv0(x)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        return x
# -------------------------------------------------







# devel code - supress execution if this is imported as module 
if __name__ == "__main__":

    from torchinfo import summary






    model_enc = EncoderGenB3blocks(n_ch_in = 3, ch = [64, 128, 256])
    model_dec = DecoderGenB3blocks(n_ch_out = 3, ch = [256, 128, 64])
    summary(model_enc, (1, 3, 128, 1152))
    summary(model_dec, (1, 256, 16, 144))








    torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
            x = self.conv2(x) # till here : pool 8
            # x = self.conv3(x) # till here : pool 16
            # x = self.conv4(x) # till here : pool 32
            # x = self.conv5(x) # till here : pool 64
            return(x)
        
    model_conv = Convnet00() 
    model_conv = model_conv.to(device)
    summary(model_conv, (1, 316, 316)) # pool 64
    summary(model_conv, (1, 156, 156)) # pool 32
    summary(model_conv, (1, 76, 76)) # pool 16
    summary(model_conv, (1, 36, 36)) # pool 8




