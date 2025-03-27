
#----------------------
#
#
#----------------------

import numpy as np
import pandas as pd
import os 
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import datetime

path_features = "C:/xc_real_projects/features"

path_clust_images = "C:/xc_real_projects/clusters_01"

path_features_full = os.path.join(path_features, 'features_medium20250319_210308.npz')

if not os.path.exists(path_clust_images):
    os.makedirs(path_clust_images)

# centf01 = 4
# centf02 = 12
# centf01 = 3
# centf02 = 13

par = {
    'centf01' : 3,
    'centf02' : 13,
    'aaa' : 333,
    }


#-------------------------
# (x) Load AEC features 
#-------------------------

data = np.load(file = path_features_full)
feat    = data['feat']
imfiles = data['imfiles']
feat.shape
imfiles.shape

# shuffle
feat, imfiles = shuffle(feat, imfiles)
feat.shape
imfiles.shape


#-------------------------
# (x) exclude time-edge bins
#-------------------------
feat.shape
feat = feat[:, :,par['centf01']:par['centf02']] 
feat.shape


#-------------------------
# (x) pooling over time to get classic feature vector 
#-------------------------

# feature_mat = np.concatenate([feat.max(2),  feat.mean(2) , feat.std(2)], axis = 1) # default
feature_mat = np.concatenate([feat.mean(2) , feat.std(2)], axis = 1) # also good 

feature_mat.shape


#-------------------------
# (x) standardize
#-------------------------
scaler = StandardScaler()
scaler.fit(feature_mat)
feature_mat_scaled_1 = scaler.transform(feature_mat)
# feature_mat_scaled_1.mean(0)
# feature_mat_scaled_1.std(0)


#-------------------------
# (x) save 
#-------------------------
tstmp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
# got to 16 bit float to have small e npz files ()
feature_mat_scaled_1 = feature_mat_scaled_1.astype('float16')
path_save_mini = os.path.join(path_features, 'features_sr' + tstmp + '.npz')
np.savez(file = path_save_mini, feat=feature_mat_scaled_1, imfiles=imfiles)





# # select valid features 
# feature_mat.std(0).shape
# sel_feats = feature_mat.std(0) != 0.000 # or feature_mat.std(0) != 0.0
# sel_feats.sum()
# feature_mat_red = feature_mat[:,sel_feats]
# feature_mat_red.shape



