#--------------------------------
# A collection of two classes from nn.Module to initialize convolutional encoders and decoders.
# The __init__ method has several parameter for fine control of the architecture
#--------------------------------

import torch
import torch.nn as nn

from torchsummary import summary
# pip install torchinfo
from torchinfo import summary






class EncoderLSTM_A(nn.Module):
    def __init__(self):
        super(EncoderLSTM_A, self).__init__()
        self.lstm0 = nn.Sequential(
            nn.LSTM(input_size = 128, hidden_size = 32, num_layers=1, bidirectional=True, batch_first=True)
        )
    def forward(self, x):
        x = self.lstm0(x)
        return(x)

model_enc = EncoderLSTM_A()

x = torch.randn(1,1152, 128)
out = model_enc(x) # works
out[0].shape

summary(model_enc, input_size=(5, 1, 1152, 128))
# summary(model_enc, (1152, 128), device = "CPU")



