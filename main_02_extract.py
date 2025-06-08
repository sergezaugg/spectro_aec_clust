#--------------------------------
# Author : Serge Zaugg
# Description : 
#--------------------------------

import os
import json
import numpy as np
import torch
from utils import encoder_based_feature_extraction, dim_reduce, evaluate_reconstruction_on_examples
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




#----------------------
# (2) evaluate reconstruction
imgpath ="D:/xc_real_projects/xc_sw_europe/xc_spectrograms"
path_trained_models = "D:/xc_real_projects/trained_models"
tstmp = '20250607_173742'
evaluate_reconstruction_on_examples(path_images = imgpath, path_models = path_trained_models, time_stamp_model = tstmp, n_images = 32)


#----------------------------------------------------------------------
# (3) extract 
# path_images = "D:/xc_real_projects/xc_parus_01/xc_spectrograms"
path_images = "D:/xc_real_projects/xc_sw_europe/xc_spectrograms"

path_enc = "D:/xc_real_projects/trained_models/20250607_173742_encoder_model_gen_B2.pth"

di = encoder_based_feature_extraction(path_enc, path_images, devel = False)
di['feature_array'].shape
di['image_file_name_array'].shape

# save as npz
tag = '_'.join(os.path.basename(path_enc).split('_')[0:2])     
out_name = os.path.join(os.path.dirname(path_images), 'full_features_' + 'saec_' + tag + '.npz')
np.savez(file = out_name, X = di['feature_array'], N = di['image_file_name_array'])










#----------------------------------------------------------------------
# (4) dim reduce 
# npzfile_full_path = "D:/xc_real_projects/xc_parus_01/full_features_saec_20250607_173742.npz"
npzfile_full_path = "D:/xc_real_projects/xc_sw_europe/full_features_saec_20250607_173742.npz"
file_name_in = os.path.basename(npzfile_full_path)

# n neighbors of UMAP currently fixed to 10 !!!!
n_neigh = 10

# load full features 
npzfile = np.load(npzfile_full_path)
X = npzfile['X']
N = npzfile['N']
X.shape
N.shape

# combine information over time
# cutting time edges (currently hard coded to 20% on each side)
ecut = np.ceil(0.10 * X.shape[2]).astype(int)
X = X[:, :, ecut:(-1*ecut)] 
print('NEW - After cutting time edges:', X.shape)
# full average pool over time 
X_mea = X.mean(axis=2)
X_std = X.std(axis=2)
X_mea.shape
X_std.shape
X = np.concatenate([X_mea, X_std], axis = 1)
print('After average/std pool along time:', X.shape)
X.shape
N.shape

# make 2d feats needed for plot 
X_2D  = dim_reduce(X, n_neigh, 2)
for n_dims_red in [2,4,8,16, 32]:
    X_red = dim_reduce(X, n_neigh, n_dims_red)
    print(X.shape, X_red.shape, X_2D.shape, N.shape)
    # save as npz
    tag_dim_red = "dimred_" + str(n_dims_red) + "_neigh_" + str(n_neigh) + "_"
    file_name_out = tag_dim_red + '_'.join(file_name_in.split('_')[2:5])
    out_name = os.path.join(os.path.dirname(npzfile_full_path), file_name_out)
    np.savez(file = out_name, X_red = X_red, X_2D = X_2D, N = N)


