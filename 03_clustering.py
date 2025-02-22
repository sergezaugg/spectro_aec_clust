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


imgpath =   "C:/xc_real_projects/xc_aec_project_n_europe/downloaded_data_img_24000sps"


model_path = "C:/xc_real_projects/models"
model_file_names = "encoder_model_epo_1_nlat_256_nblk_4.pth"

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



impsha = (128, 64)
latsha = 256
n_blck = 4

model_enc = Encoder(n_ch_in = 1, 
                    n_ch_latent=latsha, 
                    shape_input = impsha, 
                    n_conv_blocks = n_blck,
                    ch = [16, 32, 64, 256, 512],
                    po = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
                    ) 

model_enc.load_state_dict(torch.load(os.path.join(model_path, model_file_names), weights_only=True))
model_enc = model_enc.to(device)
model_enc.eval()







train_dataset = SpectroImageDataset(imgpath)
train_dataset.__len__()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512,  shuffle=True)



# extract features (by batches)
feat = []
imfiles = []
for i, (data, fi) in enumerate(train_loader, 0):    
    # print(data.shape)
    data = data.to(device)
    # data.shape
    # data.dtype
    encoded = model_enc(data).detach().cpu().numpy()
    encoded.shape
    feat.append(encoded)
    imfiles.append(fi)
    print(len(imfiles))

feat = np.concatenate(feat)
feat.shape

imfiles = np.concatenate(imfiles)
imfiles.shape






################################
# CLUSTERING

# rename
sel_subset = 50000
feature_mat = feat[0:sel_subset]
finle_name_arr = imfiles[0:sel_subset]
pa = imgpath


feature_mat.shape
feature_mat.min()
feature_mat.max()


# select valid features 
feature_mat.std(0).shape
sel_feats = feature_mat.std(0) != 0.000 # or feature_mat.std(0) != 0.0
sel_feats.sum()
feature_mat_red = feature_mat[:,sel_feats]


# standardize 
scaler = StandardScaler()
scaler.fit(feature_mat_red)
feature_mat_scaled = scaler.transform(feature_mat_red)
feature_mat_scaled.shape
feature_mat_scaled.mean(0).shape
feature_mat_scaled.mean(0)
feature_mat_scaled.std(0)


# clustering 

# clustering 
clu = AgglomerativeClustering(n_clusters=30, metric='euclidean', linkage='average')

# Manhattan distance or cosine 
# euclidean
# {'rogerstanimoto', 'dice', 'cityblock', 'matching', 'l2', 'russellrao', 
#  'mahalanobis', 'euclidean', 'minkowski', 'nan_euclidean', 'wminkowski', 'precomputed',
#    'jaccard', 'sokalsneath', 'sokalmichener', 'correlation', 'hamming', 'sqeuclidean', 
#    'cosine', 'seuclidean', 'chebyshev', 'yule', 'l1', 'braycurtis', 'haversine', 
#    'canberra', 'manhattan'}

clu = DBSCAN(eps= 10.5, min_samples=5, metric='euclidean') # eps 10.5 11.0 good min_samples=10

cluster_ids = clu.fit_predict(feature_mat_scaled)
cluster_ids.shape
pd.Series(cluster_ids).value_counts()





# finle_name_arr.shape
# cluster_ids.shape

df = pd.DataFrame({
    'file_name' :finle_name_arr,
    'cluster_id' :cluster_ids,
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
    if r['cluster_id'] == 2:
        continue
    print(r['cluster_id'])

    path_cli=  os.path.join(path_clusters, str(r['cluster_id']))
    if not os.path.exists(path_cli):
        os.mkdir(path_cli)

    src = os.path.join(pa, r['file_name'])
    dst = os.path.join(path_cli, r['newname'])
    shutil.copy(src, dst)




