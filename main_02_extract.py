#--------------------------------
# Author : Serge Zaugg
# Description : 
#--------------------------------

import torch
from utils import AutoencoderExtract
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path_images = "D:/xc_real_projects/xc_sw_europe/xc_spectrograms"
# path_images = "D:/xc_real_projects/xc_parus_01/xc_spectrograms"
# time_stamp_model = '20250613_103345' 
time_stamp_model = '20250613_103725' 

# Initialize a AEC-extractor instance
ae = AutoencoderExtract(path_images, time_stamp_model, device = device)
# evaluate reconstruction
ae.evaluate_reconstruction_on_examples(n_images = 32, shuffle = False).show()
# extract (will save to disk as npz)
ae.encoder_based_feature_extraction(devel = True)
# time_pool_and_dim_reduce
ae.time_pool_and_dim_reduce(n_neigh = 10, reduced_dim = [2, 4, 8, 16, 32])



