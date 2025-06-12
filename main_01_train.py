#--------------------------------
# Author : Serge Zaugg
# Description : Train a spectrogram auto encoder
#--------------------------------

import torch
from utils import AutoencoderTrain
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize a AEC-trainer instance with params from a json
at = AutoencoderTrain(sess_json = 'sess_01_randinit.json', device = device)
# Directly check if data augmentation as intended
at.make_data_augment_examples().show()
# Start training (.pth files will be saved to disk)
at.train_autoencoder(devel = True)




