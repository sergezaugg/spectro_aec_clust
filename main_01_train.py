#--------------------------------
# Author : Serge Zaugg
# Description : Train a spectrogram auto encoder
#--------------------------------

import torch
from utils import AutoencoderTrain
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Either, initialize a AEC-trainer with a naive model 
at = AutoencoderTrain(sess_json = 'sess_01_randinit.json', device = device)
# Or, initialize a AEC-trainer with a pre-trained model
at = AutoencoderTrain(sess_json = 'sess_02_resume.json', device = device)
# Directly check data augmentation
at.make_data_augment_examples().show()
# Start training (.pth files will be saved to disk)
at.train_autoencoder(devel = True)






