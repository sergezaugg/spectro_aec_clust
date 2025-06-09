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

# from prepare_models.model_collection import EncoderGenB2, DecoderGenB2
# save_file_name = "_gen_B2"
# model_enc = EncoderGenB2(n_ch_in = 3)
# model_dec = DecoderGenB2(n_ch_out= 3)
# model_enc = model_enc.to(device)
# model_dec = model_dec.to(device)
# summary(model_enc, (3, 128, 1152))
# summary(model_dec, (256, 1, 144))

# from prepare_models.model_collection import EncoderGenB0, DecoderGenB0
# save_file_name = "_gen_B0"
# model_enc = EncoderGenB0(n_ch_in = 3)
# model_dec = DecoderGenB0(n_ch_out = 3)
# model_enc = model_enc.to(device)
# model_dec = model_dec.to(device)
# summary(model_enc, (3, 128, 1152))
# summary(model_dec, (256, 1, 36))

# from prepare_models.model_collection import EncoderGenB1, DecoderGenB1
# save_file_name = "_gen_B1"
# model_enc = EncoderGenB1(n_ch_in = 3)
# model_dec = DecoderGenB1(n_ch_out = 3)
# model_enc = model_enc.to(device)
# model_dec = model_dec.to(device)
# summary(model_enc, (3, 128, 1152))
# summary(model_dec, (256, 1, 72))

from prepare_models.model_collection import EncoderGenB0L, DecoderGenB0L
save_file_name = "_gen_B0L"
model_enc = EncoderGenB0L(n_ch_in = 3)
model_dec = DecoderGenB0L(n_ch_out = 3)
model_enc = model_enc.to(device)
model_dec = model_dec.to(device)
summary(model_enc, (3, 128, 1152))
summary(model_dec, (512, 1, 36))


# save for later use 
torch.save(model_enc, os.path.join(path_untrained_models, 'cold_encoder' + save_file_name + '.pth'))
torch.save(model_dec, os.path.join(path_untrained_models, 'cold_decoder' + save_file_name + '.pth'))
















