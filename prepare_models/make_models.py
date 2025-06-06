#----------------------
# Author : Serge Zaugg
# Description : Create a freezed version of an AEC architecture and save to file 
#----------------------

import torch
from torchsummary import summary
import os 
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path_untrained_models = "D:/xc_real_projects/untrained_models"

# load untrained models 
from prepare_models.model_collection import EncoderGenB0, DecoderGenB0, EncoderGenB1, DecoderGenB1, EncoderGenB2, DecoderGenB2, EncoderGenB21, DecoderGenB21

# save_file_name = "_gen_B0"
# model_enc = EncoderGenB0()
# model_dec = DecoderGenB0()

# save_file_name = "_gen_B1"
# model_enc = EncoderGenB1()
# model_dec = DecoderGenB1()

# save_file_name = "_gen_B2"
# model_enc = EncoderGenB2()
# model_dec = DecoderGenB2()

save_file_name = "_gen_B21"
model_enc = EncoderGenB21()
model_dec = DecoderGenB21()


model_enc = model_enc.to(device)
model_dec = model_dec.to(device)

# check architecture 
summary(model_enc, (1, 128, 1152))
# summary(model_dec, (128, 1, 36))
# summary(model_dec, (128, 1, 72))
summary(model_dec, (64, 1, 144))

# save for later use 
torch.save(model_enc, os.path.join(path_untrained_models, 'cold_encoder' + save_file_name + '.pth'))
torch.save(model_dec, os.path.join(path_untrained_models, 'cold_decoder' + save_file_name + '.pth'))
















