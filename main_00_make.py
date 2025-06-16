#----------------------
# Author : Serge Zaugg
# Description : Create a frozen versions of AEC architectures and save to file (run only once)
#----------------------

import torch
from torchinfo import summary
import os 
import yaml

# load path from config 
with open('./config/config.yaml') as f:
    conf = yaml.safe_load(f)
path_untrained_models = conf['path_untrained_models']

from model_collection.model_collection import EncoderGenBTP08 as Encoder
from model_collection.model_collection import DecoderGenBTP08 as Decoder
save_file_name = "_GenBTP08_CH0256"
model_enc = Encoder(n_ch_in = 3, ch = [64, 128, 128, 128, 256])
model_dec = Decoder(n_ch_out = 3, ch = [256, 128, 128, 128, 64])
summary(model_enc, (1, 3, 128, 1152))
summary(model_dec, (1, 256, 1, 144))
torch.save(model_enc, os.path.join(path_untrained_models, 'cold_encoder' + save_file_name + '.pth'))
torch.save(model_dec, os.path.join(path_untrained_models, 'cold_decoder' + save_file_name + '.pth'))


from model_collection.model_collection import EncoderGenBTP16 as Encoder
from model_collection.model_collection import DecoderGenBTP16 as Decoder
save_file_name = "_GenBTP16_CH0256"
model_enc = Encoder(n_ch_in = 3, ch = [64, 128, 128, 128, 256])
model_dec = Decoder(n_ch_out = 3, ch = [256, 128, 128, 128, 64])
summary(model_enc, (1, 3, 128, 1152))
summary(model_dec, (1, 256, 1, 72))
torch.save(model_enc, os.path.join(path_untrained_models, 'cold_encoder' + save_file_name + '.pth'))
torch.save(model_dec, os.path.join(path_untrained_models, 'cold_decoder' + save_file_name + '.pth'))


from model_collection.model_collection import EncoderGenBTP32 as Encoder
from model_collection.model_collection import DecoderGenBTP32 as Decoder
save_file_name = "_GenBTP32_CH0256"
model_enc = Encoder(n_ch_in = 3, ch = [64, 128, 128, 128, 256])
model_dec = Decoder(n_ch_out = 3, ch = [256, 128, 128, 128, 64])
summary(model_enc, (1, 3, 128, 1152))
summary(model_dec, (1, 256, 1, 36))
torch.save(model_enc, os.path.join(path_untrained_models, 'cold_encoder' + save_file_name + '.pth'))
torch.save(model_dec, os.path.join(path_untrained_models, 'cold_decoder' + save_file_name + '.pth'))


from model_collection.model_collection import EncoderGenBTP32 as Encoder
from model_collection.model_collection import DecoderGenBTP32 as Decoder
save_file_name = "_GenBTP32_CH0512"
model_enc = Encoder(n_ch_in = 3, ch = [64, 128, 128, 256, 512])
model_dec = Decoder(n_ch_out = 3, ch = [512, 256, 128, 128, 64])
summary(model_enc, (1, 3, 128, 1152), depth = 1)
summary(model_dec, (1, 512, 1, 36), depth = 1)
torch.save(model_enc, os.path.join(path_untrained_models, 'cold_encoder' + save_file_name + '.pth'))
torch.save(model_dec, os.path.join(path_untrained_models, 'cold_decoder' + save_file_name + '.pth'))


from model_collection.model_collection import EncoderGenBTP64 as Encoder
from model_collection.model_collection import DecoderGenBTP64 as Decoder
save_file_name = "_GenBTP64_CH0256"
model_enc = Encoder(n_ch_in = 3, ch = [64, 128, 128, 128, 256])
model_dec = Decoder(n_ch_out = 3, ch = [256, 128, 128, 128, 64])
summary(model_enc, (1, 3, 128, 1152))
summary(model_dec, (1, 256, 1, 18))
torch.save(model_enc, os.path.join(path_untrained_models, 'cold_encoder' + save_file_name + '.pth'))
torch.save(model_dec, os.path.join(path_untrained_models, 'cold_decoder' + save_file_name + '.pth'))


from model_collection.model_experiments import EncoderGenCTP128 as Encoder
from model_collection.model_experiments import DecoderGenCTP128 as Decoder
save_file_name = "_GenC_experiment"
model_enc = Encoder(n_ch_in = 3, n_ch_out = 256, ch = [64, 128, 128, 256])
model_dec = Decoder(n_ch_in = 256, n_ch_out = 3, ch = [256, 128, 128, 64])
summary(model_enc, (1, 3, 128, 1152), depth = 1)
summary(model_dec, (1, 256, 1, 36), depth = 1)
torch.save(model_enc, os.path.join(path_untrained_models, 'cold_encoder' + save_file_name + '.pth'))
torch.save(model_dec, os.path.join(path_untrained_models, 'cold_decoder' + save_file_name + '.pth'))



from model_collection.model_collection import EncoderGenB3blocks as Encoder
from model_collection.model_collection import DecoderGenB3blocks as Decoder
save_file_name = "_GenB3blocks"
model_enc = Encoder(n_ch_in = 3, ch = [64, 128, 256])
model_dec = Decoder(n_ch_out = 3, ch = [256, 128, 64])
summary(model_enc, (1, 3, 128, 1152), depth = 1)
summary(model_dec, (1, 256, 16, 144), depth = 1)
torch.save(model_enc, os.path.join(path_untrained_models, 'cold_encoder' + save_file_name + '.pth'))
torch.save(model_dec, os.path.join(path_untrained_models, 'cold_decoder' + save_file_name + '.pth'))


