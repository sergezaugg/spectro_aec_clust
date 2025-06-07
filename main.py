#--------------------------------
# Author : Serge Zaugg
# Description : 
#--------------------------------

import os
import json
import numpy as np
import torch
from torchsummary import summary
from utils import SpectroImageDataset, make_data_augment_examples, get_models, train_autoencoder
from utils import evaluate_reconstruction_on_examples, encoder_based_feature_extraction, dim_reduce
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#----------------------------------------------------------------------
# (1) train 
with open(os.path.join('./session_params', 'sess_01.json' )) as f:
    sess_info = json.load(f)

train_dataset = SpectroImageDataset(sess_info['imgpath_train'], par = sess_info['data_generator'], augment_1 = True, denoise_1 = False, augment_2 = False, denoise_2 = True)
test_dataset  = SpectroImageDataset(sess_info['imgpath_test'],  par = sess_info['data_generator'], augment_1 = False, denoise_1 = False, augment_2 = False, denoise_2 = True)
make_data_augment_examples(pt_dataset = train_dataset, batch_size = 16).show()
model_enc, model_dec = get_models(sess_info)
summary(model_enc, (3, 128, 1152))
summary(model_dec, (256, 1, 144))
train_autoencoder(sess_info, train_dataset, test_dataset, model_enc, model_dec, devel = False)

#----------------------------------------------------------------------
# (2) evaluate 
imgpath ="D:/xc_real_projects/xc_sw_europe/xc_spectrograms"
n_images = 32
path_trained_models = "D:/xc_real_projects/trained_models"
tstmp = '20250607_112041'
# "D:\xc_real_projects\trained_models\20250607_112041_encoder_model_gen_B2.pth"
evaluate_reconstruction_on_examples(path_images = imgpath, path_models = path_trained_models, time_stamp_model = tstmp, n_images = 32)

#----------------------------------------------------------------------
# (3) extract 
# path_images = "D:/xc_real_projects/xc_sw_europe/images_24000sps_20250406_092522"
path_images = "D:/xc_real_projects/xc_parus_01/xc_spectrograms"

path_enc = "D:/xc_real_projects/trained_models/20250607_112041_encoder_model_gen_B2.pth"



di = encoder_based_feature_extraction(path_enc, path_images, devel = True)
di['feature_array'].shape
di['image_file_name_array'].shape

# save as npz

tag = '_'.join(os.path.basename(path_enc).split('_')[0:2])
         
# tstmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
out_name = os.path.join(os.path.dirname(path_images), 'full_features_' + 'saec_' + tag + '.npz')
np.savez(file = out_name, X = di['feature_array'], N = di['image_file_name_array'])










#----------------------------------------------------------------------
# (4) dim reduce 

# npzfile_full_path = "D:/xc_real_projects/xc_parus_01/full_features_saec_20250607_083009.npz"
npzfile_full_path = "D:/xc_real_projects/xc_parus_01/full_features_saec_20250607_112041.npz"
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
ecut = np.ceil(0.20 * X.shape[2]).astype(int)
X = X[:, :, ecut:(-1*ecut)] 
print('NEW - After cutting time edges:', X.shape)
# full average pool over time 
X = X.mean(axis=2)
print('After average pool along time:', X.shape)

X.shape
N.shape

# make 2d feats needed for plot 
X_2D  = dim_reduce(X, n_neigh, 2)
for n_dims_red in [2,4,8,16]:
    X_red = dim_reduce(X, n_neigh, n_dims_red)
    print(X.shape, X_red.shape, X_2D.shape, N.shape)
    # save as npz
    tag_dim_red = "dimred_" + str(n_dims_red) + "_neigh_" + str(n_neigh) + "_"
    file_name_out = tag_dim_red + '_'.join(file_name_in.split('_')[2:5])
    out_name = os.path.join(os.path.dirname(npzfile_full_path), file_name_out)
    np.savez(file = out_name, X_red = X_red, X_2D = X_2D, N = N)


