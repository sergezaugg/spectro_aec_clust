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

path_models = "D:/xc_real_projects/trained_models"
# time_stamp_model = '20250608_155900' # gen_B0 old training set
# time_stamp_model = '20250609_025355' # gen_B0 new big training set
# time_stamp_model = '20250609_135101' # gen_B1 new big training set
time_stamp_model = '20250610_002647' # gen_B0L (large) new big training set




# Initialize a AEC-extractor instance
ae = AutoencoderExtract(path_images, path_models, time_stamp_model)

# evaluate reconstruction
ae.evaluate_reconstruction_on_examples(n_images = 32, shuffle = True).show()

# extract (will save to disk as npz in parent dir of path_models)
ae.encoder_based_feature_extraction(devel = False)

# time_pool_and_dim_reduce
ae.time_pool_and_dim_reduce(n_neigh = 10, reduced_dim = [2, 4, 8, 16, 32])



