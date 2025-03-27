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

path_features_full = os.path.join(path_features, 'features_medium20250319_210308.npz')
#                                                 features_medium20250319_210308.npz

par = {
    'centf01' : 7,
    'centf02' : 9,
    }

#-------------------------
# Load AEC features 
data = np.load(file = path_features_full)
feat    = data['feat']
imfiles = data['imfiles']
feat.shape
imfiles.shape

#-------------------------
# shuffle
feat, imfiles = shuffle(feat, imfiles)
feat.shape
imfiles.shape

#-------------------------
# exclude time-edge bins
feat.shape
feat = feat[:, :,par['centf01']:par['centf02']] 
feat.shape

#-------------------------
# pooling over time to get classic feature vector 
feat.max(2).shape
# feature_mat = np.concatenate([feat.max(2),  feat.mean(2) , feat.std(2)], axis = 1) # default
feature_mat = np.concatenate([feat.mean(2) , feat.std(2)], axis = 1) # default

#-------------------------
# standardize
scaler = StandardScaler()
scaler.fit(feature_mat)
feature_mat_scaled = scaler.transform(feature_mat)
feature_mat_scaled.shape

#-------------------------
# save 
tstmp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
# got to 16 bit float to have small e npz files ()
feature_mat_scaled = feature_mat_scaled.astype('float16')
path_save_mini = os.path.join(path_features, 'feat' + tstmp + '_' + str(par['centf01']) + '_' + str(par['centf02']) + '.npz')
np.savez(file = path_save_mini, feat=feature_mat_scaled, imfiles=imfiles)






