#----------------------
#
#
#----------------------

import numpy as np
import torch
import plotly.express as px
import numpy as np
from torchsummary import summary
import os 
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from ptutils import SpectroImageDataset, Encoder, Decoder



imgpath = "xxxxxxxxxxxxxxxx"
model_path = "xxxxxxxxxxxxxxxxxxx"



torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load the model
model_enc = Encoder(embedding_dim = 512, channels = 3)
model_enc.load_state_dict(torch.load(os.path.join(model_path, "encoder_model.pth"), weights_only=True))
model_enc = model_enc.to(device)
model_enc.eval()




train_dataset = SpectroImageDataset(imgpath)
train_dataset.__len__()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32,  shuffle=True)



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



type(fi)



################################
# AGGLOM CLUSTERING

# rename
feature_mat = feat
finle_name_arr = imfiles
pa = imgpath

# imgpath
# imfiles

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import shutil
import os
from sklearn.cluster import DBSCAN


# clustering 
clu = AgglomerativeClustering(n_clusters=21, metric='euclidean', linkage='average')

# clu = DBSCAN(eps= 5, min_samples=5, metric='euclidean')

# clustering = clu.fit(feature_mat)
cluster_ids = clu.fit_predict(feature_mat)
cluster_ids.shape
pd.Series(cluster_ids).value_counts()

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
    print(r)

    path_cli=  os.path.join(path_clusters, str(r['cluster_id']))
    if not os.path.exists(path_cli):
        os.mkdir(path_cli)

    src = os.path.join(pa, r['file_name'])
    dst = os.path.join(path_cli, r['newname'])
    shutil.copy(src, dst)




