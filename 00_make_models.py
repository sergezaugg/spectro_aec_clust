#----------------------
# Author : Serge Zaugg
# Description : Creat a freezed version of an AEC architecture and save to file 
#----------------------

import torch
from torchsummary import summary
import os 
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#----------------------
# load untrained models 
from custom_models import EncoderGenA, DecoderGenA
from custom_models import EncoderGenB, DecoderGenB

model_enc = EncoderGenA()
model_dec = DecoderGenA()

# model_enc = EncoderGenB()
# model_dec = DecoderGenB()

model_enc = model_enc.to(device)
model_dec = model_dec.to(device)

#----------------------
# check architecture 
summary(model_enc, (1, 128, 1152))
summary(model_dec, (128, 1, 36))
summary(model_dec, (128, 1, 72))

#----------------------
# save for later use 
path_untrained_models = "D:/xc_real_projects/untrained_models"

save_file_name = "_gen_A"

torch.save(model_enc, os.path.join(path_untrained_models, 'encoder' + save_file_name + '.pth'))
torch.save(model_dec, os.path.join(path_untrained_models, 'decoder' + save_file_name + '.pth'))





















