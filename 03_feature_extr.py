#----------------------
# Author : Serge Zaugg
# Description : 
#----------------------

import numpy as np
import os 
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import datetime
import torch
from custom_models import SpectroImageDataset, EncoderAvgpool
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define paths
model_path = "C:/xc_real_projects/models/encoder_model_20250319_210308_epo_20.pth"

path_xc = "C:/xc_real_projects/xc_parus_01/"

imgpath = os.path.join(path_xc, "downloaded_data_img_24000sps")


# load a trained encoder 
model_enc = EncoderAvgpool()
model_enc.load_state_dict(torch.load(os.path.join(model_path), weights_only=True))
model_enc = model_enc.to(device)
_ = model_enc.eval()

# prepare dataloader
test_dataset = SpectroImageDataset(imgpath, edge_attenuation = False)
test_dataset.__len__()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

# extract features (by batches)
feat_li = []
imfiles = []
for i, (data, _, fi) in enumerate(test_loader, 0):    
    print(data.shape)
    data = data.to(device)
    encoded = model_enc(data).detach().cpu().numpy()
    encoded.shape
    feat_li.append(encoded)
    imfiles.append(fi)
    print(len(imfiles))
    # if i > 2:
    #     break

# transform lists to array 
feat = np.concatenate(feat_li)
feat = feat.squeeze()
imfiles = np.concatenate(imfiles)

# shuffle
feat, imfiles = shuffle(feat, imfiles)
feat.shape
imfiles.shape

# save all relevant objects 
tstmp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
features_save_path = os.path.join(path_xc, "features" + tstmp) 
if not os.path.exists(features_save_path):
    os.makedirs(features_save_path)
path_save_npz = os.path.join(features_save_path, 'features_from_encoder' + tstmp + '.npz')
encoder_id = np.array(model_path)
np.savez(file = path_save_npz, feat=feat, imfiles=imfiles, encoder_id = encoder_id)



#----------------------
# Time-pooling 

edge_cut_li = [[0,16], [2,14], [4,12], [6,10]]

for edgcut in edge_cut_li:
    print(edgcut)
  
    #-------------------------
    # exclude time-edge bins
    feat_cut = feat[:, :, edgcut[0]:edgcut[1]] 
    feat_cut.shape

    #-------------------------
    # pooling over time to get classic feature vector 
    feat_cut.max(2).shape
    # feature_mat = np.concatenate([feat_cut.max(2),  feat_cut.mean(2) , feat_cut.std(2)], axis = 1) # default
    feature_mat = np.concatenate([feat_cut.mean(2) , feat_cut.std(2)], axis = 1) # default

    #-------------------------
    # standardize
    scaler = StandardScaler()
    scaler.fit(feature_mat)
    feature_mat_scaled = scaler.transform(feature_mat)
    feature_mat_scaled.shape

    #-------------------------
    # save 
    feature_mat_scaled = feature_mat_scaled.astype('float16') # got to 16 bit float to have small e npz files ()
    path_save_mini = os.path.join(features_save_path, 'features_timepooled_' + str(edgcut[0]) + '_' + str(edgcut[1]) + '.npz')
    np.savez(file = path_save_mini, feat=feature_mat_scaled, imfiles=imfiles)





