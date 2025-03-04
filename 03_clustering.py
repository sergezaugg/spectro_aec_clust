#----------------------
#
#
#----------------------

import numpy as np
import pandas as pd
import torch
import plotly.express as px
import os 
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
import shutil
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from custom_models import Encoder, Decoder, SpectroImageDataset
import pickle


torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


imgpath =   "C:/xc_real_projects/xc_aec_project_n_europe/downloaded_data_img_24000sps"

model_path = "C:/xc_real_projects/models"



tstmp = "20250304_182800"
epotag = '_epo_5'
path_enc = 'encoder_model_' + tstmp + epotag + '.pth'
path_dec = 'decoder_model_' + tstmp + epotag + '.pth'
path_par = 'params_model_'  + tstmp + epotag + '.json'


with open(os.path.join(model_path, path_par), 'rb') as fp:
    par = pickle.load(fp)


model_enc = Encoder(n_ch_in = par['e']['n_ch_in'], 
                    ch = par['e']['ch'],
                    po = par['e']['po']
                    ) 
model_enc.load_state_dict(torch.load(os.path.join(model_path, path_enc), weights_only=True))
model_enc = model_enc.to(device)
_ = model_enc.eval()
















train_dataset = SpectroImageDataset(imgpath)
train_dataset.__len__()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=500,  shuffle=True)



# extract features (by batches)
feat_li = []
imfiles = []
for i, (data, data_augm, fi) in enumerate(train_loader, 0):    
    # print(data.shape)
    data = data.to(device)
    # data.shape
    # data.dtype
    encoded = model_enc(data).detach().cpu().numpy()
    encoded.shape
    feat_li.append(encoded)
    imfiles.append(fi)
    print(len(imfiles))

feat = np.concatenate(feat_li)
feat = feat.squeeze()
feat.shape

imfiles = np.concatenate(imfiles)
imfiles.shape



# taka function overt time to get classic feature vector 
feat_max = feat.max(2)
feat_mea = feat.mean(2)
feat_max.shape
feat_mea.shape


################################
# CLUSTERING

# rename
sel_subset = 50000
feature_mat = feat_max[0:sel_subset]
# feature_mat = feat_mea[0:sel_subset]
finle_name_arr = imfiles[0:sel_subset]
pa = imgpath
feature_mat.shape



# select valid features 
feature_mat.std(0).shape
sel_feats = feature_mat.std(0) != 0.000 # or feature_mat.std(0) != 0.0
sel_feats.sum()
feature_mat_red = feature_mat[:,sel_feats]

feature_mat_red.shape

# standardize 
scaler = StandardScaler()
scaler.fit(feature_mat_red)
feature_mat_scaled = scaler.transform(feature_mat_red)
# feature_mat_scaled.shape
# feature_mat_scaled.mean(0).shape
# feature_mat_scaled.mean(0)
# feature_mat_scaled.std(0)


# clustering 

# # clustering 
# clu = AgglomerativeClustering(n_clusters=30, metric='euclidean', linkage='single')
# cluster_ids = clu.fit_predict(feature_mat_scaled)
# cluster_ids.shape
# pd.Series(cluster_ids).value_counts()




for eps_i in range(18,30,):
    print("-----------")
    print(">> eps_i", eps_i)
    clu = DBSCAN(eps= eps_i, min_samples=5, metric='euclidean') # eps 10.5 11.0 good min_samples=10
    cluster_ids = clu.fit_predict(feature_mat_scaled)
    cluster_ids.shape
    pd.Series(cluster_ids).value_counts()[0:10]
    print("")
    print("")
    print("")


clu = DBSCAN(eps= 22, min_samples=1, metric='euclidean') # eps 10.5 11.0 good min_samples=10
cluster_ids = clu.fit_predict(feature_mat_scaled)
cluster_ids.shape
pd.Series(cluster_ids).value_counts()[0:20]


#selec tonly large enought clustes 
sel = pd.Series(cluster_ids).value_counts() > 6

sel2 = pd.Series(cluster_ids).value_counts().loc[sel].index
cluster_ids.shape
sel.shape

cluster_ids_sel = cluster_ids[ pd.Series(cluster_ids).isin(sel2)]
finle_name_arr_sel = finle_name_arr[ pd.Series(cluster_ids).isin(sel2)]
cluster_ids_sel.shape
finle_name_arr_sel.shape



# save images by cluster id 
df = pd.DataFrame({
    'file_name' :finle_name_arr_sel,
    'cluster_id' :cluster_ids_sel,
    })
df['newname'] = df['cluster_id'].astype(str).str.cat(others=df['file_name'], sep='_')
# df['file_name'][3]
path_clusters = os.path.join(os.path.dirname(pa), 'clusters')
if not os.path.exists(path_clusters):
    os.mkdir(path_clusters)
for i,r in df.iterrows():
    # print(r)
    if r['cluster_id'] == -1:
        continue
    if r['cluster_id'] == 1:
        continue
    print(r['cluster_id'])
    path_cli=  os.path.join(path_clusters, str(r['cluster_id']))
    if not os.path.exists(path_cli):
        os.mkdir(path_cli)
    src = os.path.join(pa, r['file_name'])
    dst = os.path.join(path_cli, r['newname'])
    shutil.copy(src, dst)




