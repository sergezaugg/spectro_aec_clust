#--------------------------------
# Author : Serge Zaugg
# Description : 
#--------------------------------

# import os
# import json
# import numpy as np
import torch
from torchsummary import summary
from utils import AutoencoderTrain
from utils import evaluate_reconstruction_on_examples
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#----------------------------------------------------------------------
# (1) train 


at = AutoencoderTrain(sess_json = 'sess_01.json')

at.make_data_augment_examples().show()

summary(at.model_enc, (3, 128, 1152))

summary(at.model_dec, (256, 1, 144))

at.train_autoencoder(devel = True)


#----------------------------------------------------------------------
# (2) evaluate reconstruction
imgpath ="D:/xc_real_projects/xc_sw_europe/xc_spectrograms"
path_trained_models = "D:/xc_real_projects/trained_models"
tstmp = '20250607_173742'
evaluate_reconstruction_on_examples(path_images = imgpath, path_models = path_trained_models, time_stamp_model = tstmp, n_images = 32)



