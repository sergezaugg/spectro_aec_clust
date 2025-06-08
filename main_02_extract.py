#--------------------------------
# Author : Serge Zaugg
# Description : 
#--------------------------------

# import os
# import json
# import numpy as np
import torch
from utils import encoder_based_feature_extraction, evaluate_reconstruction_on_examples, time_pool_and_dim_reduce
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


path_images = "D:/xc_real_projects/xc_sw_europe/xc_spectrograms"
path_models = "D:/xc_real_projects/trained_models"
tstmp = '20250607_173742'

# evaluate reconstruction
evaluate_reconstruction_on_examples(path_images = path_images, path_models = path_models, time_stamp_model = tstmp, n_images = 32).show()

# extract (will save to disk as npz in parent dir of path_models)
encoder_based_feature_extraction(path_images = path_images, path_models = path_models, time_stamp_model = tstmp ,  devel = True)

# time_pool_and_dim_reduce
time_pool_and_dim_reduce(path_images, tstmp)



