#--------------------------------
# Author : Serge Zaugg
# Description : Train a spectrogram auto encoder
#--------------------------------

import torch
from torchsummary import summary
from utils import AutoencoderTrain
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize a AEC-trainer instance with params from a json
at = AutoencoderTrain(sess_json = 'sess_01.json')

# Directly check if data augmentation is as intended
at.make_data_augment_examples().show()

# Have a quick look at AEC architecture
summary(at.model_enc, (3, 128, 1152))
summary(at.model_dec, (256, 1, 144))

# Start training (.pth files will be saved to disk)
at.train_autoencoder(devel = True)




