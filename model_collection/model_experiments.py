#--------------------------------
# A collection of two classes from nn.Module to initialize convolutional encoders and decoders.
# The __init__ method has several parameter for fine control of the architecture
#--------------------------------

import torch
import torch.nn as nn
from torchinfo import summary


# -------------------------------------------------
# GEN C (freq-pool : 128 - time-pool : 128)

class EncoderGenCTP128(nn.Module):
    def __init__(self, n_ch_in = 3, n_ch_out=256,
                 ch = [64, 64, 64, 128, 128, 128]
                 ):
        super().__init__()
        po = [(4, 2), (4, 2), (2, 2), (2, 2), (2, 2)]
        self.padding =  "same"
        conv_kernel = (3,3)
        self.conv0 = nn.Sequential(
            nn.Conv2d(n_ch_in,  ch[0], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(),
            nn.Conv2d(ch[0],  ch[0], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(),
            nn.AvgPool2d(po[0], stride=po[0]))
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(),
            nn.Conv2d(ch[1], ch[1], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(),
            nn.AvgPool2d(po[1], stride=po[1]))
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch[1], ch[2], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(),
            nn.Conv2d(ch[2], ch[2], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(),
            nn.AvgPool2d(po[2], stride=po[2]))
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch[2], ch[3], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[3]),
            nn.ReLU(),
            nn.Conv2d(ch[3], ch[3], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[3]),
            nn.ReLU(),
            nn.AvgPool2d(po[3], stride=po[3]))
        self.conv4 = nn.Sequential(
            nn.Conv2d(ch[3], ch[3], kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[3]),
            nn.ReLU(),
            nn.Conv2d(ch[3], n_ch_out, kernel_size=conv_kernel, stride=1, padding=self.padding),
            nn.BatchNorm2d(n_ch_out),
            nn.ReLU(),
            nn.AvgPool2d(po[4], stride=po[4]))
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return(x)


class DecoderGenCTP128(nn.Module):
    def __init__(self, n_ch_in=256, n_ch_out=3, 
                 ch = [128, 128, 128, 64, 64, 64]):
        super().__init__()
        self.padding =  "same"
        po =  [(2, 2), (2, 2), (2, 2), (4, 2), (4, 2)]
        self.tconv0 = nn.Sequential(
            nn.Upsample(scale_factor=po[0], mode='bilinear'),
            nn.Conv2d(n_ch_in,  ch[0], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(),
            nn.Conv2d(ch[0],  ch[0], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(),
            )
        self.tconv1 = nn.Sequential(
            nn.Upsample(scale_factor=po[1], mode='bilinear'),
            nn.Conv2d(ch[0], ch[1], kernel_size=(3,3), stride=1, padding=self.padding), 
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(),
            nn.Conv2d(ch[1], ch[1], kernel_size=(3,3), stride=1, padding=self.padding), 
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(),
            )
        self.tconv2 = nn.Sequential(
            nn.Upsample(scale_factor=po[2], mode='bilinear'),
            nn.Conv2d(ch[1], ch[2], kernel_size=(3,3), stride=1, padding=self.padding),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(),
            nn.Conv2d(ch[2], ch[2], kernel_size=(3,3), stride=1, padding=self.padding),    
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(),
            )
        self.tconv3 = nn.Sequential(
            nn.Upsample(scale_factor=po[3], mode='bilinear'),
            nn.Conv2d(ch[2], ch[3], kernel_size=(3,3), stride=1, padding=self.padding), 
            nn.BatchNorm2d(ch[3]),
            nn.ReLU(),
            nn.Conv2d(ch[3], ch[3], kernel_size=(3,3), stride=1, padding=self.padding), 
            nn.BatchNorm2d(ch[3]),
            nn.ReLU(),
            )
        self.out_map = nn.Sequential(
            nn.Upsample(scale_factor=po[4], mode='bilinear'),
            nn.Conv2d(ch[3], ch[3], kernel_size=(3,3), stride=1, padding=self.padding), 
            nn.BatchNorm2d(ch[3]),
            nn.ReLU(),
            nn.Conv2d(ch[3], n_ch_out, kernel_size=(3,3), stride=1, padding=self.padding), 
            nn.BatchNorm2d(n_ch_out),
            nn.Sigmoid(),
            )

    def forward(self, x):
        x = self.tconv0(x)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.out_map(x)
        return x

# -------------------------------------------------




if __name__ == "__main__":

    model_enc = EncoderGenCTP128(n_ch_in = 3, n_ch_out = 256, ch = [64, 128, 128, 256])
    model_dec = DecoderGenCTP128(n_ch_in = 256, n_ch_out = 3, ch = [256, 128, 128, 64])
    
    summary(model_enc, (1, 3, 128, 1152), depth = 1)
    summary(model_dec, (1, 256, 1, 36), depth = 1)


    
    # # from copilot 
    # class AvgPoolUpsampleSpectrogramAutoencoder(nn.Module):
    #     def __init__(self):
    #         super(AvgPoolUpsampleSpectrogramAutoencoder, self).__init__()
    #         # Encoder with AvgPool2d
    #         self.encoder = nn.Sequential(
    #             nn.Conv2d(1, 16, kernel_size=3, padding=1),     # [B, 1, H, W] -> [B, 16, H, W]
    #             nn.ReLU(True),
    #             nn.AvgPool2d(kernel_size=2, stride=2),          # [B, 16, H/2, W/2]
    #             nn.Conv2d(16, 32, kernel_size=3, padding=1),    # [B, 32, H/2, W/2]
    #             nn.ReLU(True),
    #             nn.AvgPool2d(kernel_size=2, stride=2),          # [B, 32, H/4, W/4]
    #             nn.Conv2d(32, 64, kernel_size=3, padding=1),    # [B, 64, H/4, W/4]
    #             nn.ReLU(True),
    #             nn.AvgPool2d(kernel_size=2, stride=2)           # [B, 64, H/8, W/8]
    #         )
    #         # Decoder with nn.Upsample + nn.Conv2d
    #         self.decoder = nn.Sequential(
    #             nn.Upsample(scale_factor=2, mode='nearest'),    # [B, 64, H/4, W/4]
    #             nn.Conv2d(64, 32, kernel_size=3, padding=1),
    #             nn.ReLU(True),
    #             nn.Upsample(scale_factor=2, mode='nearest'),    # [B, 32, H/2, W/2]
    #             nn.Conv2d(32, 16, kernel_size=3, padding=1),
    #             nn.ReLU(True),
    #             nn.Upsample(scale_factor=2, mode='nearest'),    # [B, 16, H, W]
    #             nn.Conv2d(16, 1, kernel_size=3, padding=1),
    #             nn.Sigmoid()  # Use sigmoid if spectrograms are normalized to [0,1]
    #         )
















    class EncoderLSTM_A(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm0 = nn.LSTM(input_size = 128, hidden_size = 32, num_layers=1, bidirectional=True, batch_first=True)
            
        def forward(self, x):
            out, hidden = self.lstm0(x)
            return out, hidden


    class LSTMNet(nn.Module):
        def __init__(self, vocab_size=20, embed_dim=300, hidden_dim=512, num_layers=2):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.decoder = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x):
            embed = self.embedding(x)
            out, hidden = self.encoder(embed)
            out = self.decoder(out)
            out = out.view(-1, out.size(2))
            return out, hidden




    summary(
        EncoderLSTM_A(), 
        (1, 128),
        dtypes=[torch.long]
        )

    summary(
        LSTMNet(),
        (1, 100),
        dtypes=[torch.long],
        )




    model_enc = EncoderLSTM_A()

    # x = torch.randn(1,1152, 128)
    # out = model_enc(x) # works
    # out[0].shape


