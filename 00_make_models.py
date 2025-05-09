#----------------------
# Author : Serge Zaugg
# Description : 
#----------------------

# import plotly.express as px
import torch
from torchsummary import summary
import os 






from custom_models_2 import EncoderAvgpool, DecoderTranspNew


torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path_untrained_models = "D:/xc_real_projects/untrained_models"



#----------------------
# define untrained models 
model_enc = EncoderAvgpool()
model_dec = DecoderTranspNew()

model_enc = model_enc.to(device)
summary(model_enc, (1, 128, 1152))

model_dec = model_dec.to(device)
summary(model_dec, (128, 1, 36))


# encoded
torch.save(model_enc, os.path.join(path_untrained_models, 'test_xxxxxx.pth'))

model_enc_t = torch.load(os.path.join(path_untrained_models, 'test_xxxxxx.pth'), weights_only = False)















