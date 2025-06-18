#--------------------------------
# Author : Serge Zaugg
# Description : Train a spectrogram auto encoder
#--------------------------------

import torch
from utils import MakeColdAutoencoders, AutoencoderTrain
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# (run once) Create the cold (=random init) instances of the models
mca = MakeColdAutoencoders()
mod_arch = mca.make()

# Either, initialize a AEC-trainer with a naive model 
at = AutoencoderTrain(sess_json = 'sess_01_randinit.json', device = device)
# Or, initialize a AEC-trainer with a pre-trained model
at = AutoencoderTrain(sess_json = 'sess_02_resume.json', device = device)

# Directly check data augmentation
at.make_data_augment_examples().show()
# Start training (.pth files will be saved to disk)
at.train_autoencoder(devel = False)

