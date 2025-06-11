#----------------------
# Author : Serge Zaugg
# Description : Create a freezed version of an AEC architecture and save to file 
#----------------------

import torch
from torchsummary import summary
import os 

path_untrained_models = "D:/xc_real_projects/untrained_models"

# from model_collection.model_collection import EncoderGenB2 as Encoder
# from model_collection.model_collection import DecoderGenB2 as Decoder
# save_file_name = "_gen_B2"

# from model_collection.model_collection import EncoderGenB0 as Encoder
# from model_collection.model_collection import DecoderGenB0 as Decoder
# save_file_name = "_gen_B0"

# from model_collection.model_collection import EncoderGenB1 as Encoder
# from model_collection.model_collection import DecoderGenB1 as Decoder
# save_file_name = "_gen_B1"

from model_collection.model_collection import EncoderGenB0L as Encoder
from model_collection.model_collection import DecoderGenB0L as Decoder
save_file_name = "_gen_B0L"

model_enc = Encoder(n_ch_in = 3)
model_dec = Decoder(n_ch_out = 3)
summary(model_enc, (3, 128, 1152), device = "CPU")
summary(model_dec, (512, 1, 36), device = "CPU")
# save for later use 
torch.save(model_enc, os.path.join(path_untrained_models, 'cold_encoder' + save_file_name + '.pth'))
torch.save(model_dec, os.path.join(path_untrained_models, 'cold_decoder' + save_file_name + '.pth'))







# torch.cuda.is_available()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# model_enc = model_enc.to(device)
# model_dec = model_dec.to(device)











