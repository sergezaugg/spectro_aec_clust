#--------------------------------
# A collection of two classes from nn.Module to initialize convolutional encoders and decoders.
# The __init__ method has several parameter for fine control of the architecture
#--------------------------------

import torch
import torch.nn as nn
from torchinfo import summary

if __name__ == "__main__":

    # model_enc = EncoderGenCTP128(n_ch_in = 3, n_ch_out = 256, ch = [64, 128, 128, 256])
    # model_dec = DecoderGenCTP128(n_ch_in = 256, n_ch_out = 3, ch = [256, 128, 128, 64])
    # summary(model_enc, (1, 3, 128, 1152), depth = 1)
    # summary(model_dec, (1, 256, 1, 36), depth = 1)

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


