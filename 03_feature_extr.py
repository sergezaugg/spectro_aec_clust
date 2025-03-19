#----------------------
#
#
#----------------------

import numpy as np
import pandas as pd
import torch
import os 
from custom_models import SpectroImageDataset, EncoderAvgpool
import pickle

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

imgpath = "C:/xc_real_projects/xc_streamlit_sw_eur/downloaded_data_img_24000sps_rect"

path_features = "C:/xc_real_projects/features"
model_path = "C:/xc_real_projects/models"

tstmp = '20250319_210308'
epotag = '_epo_20'

model_enc = EncoderAvgpool()

path_save = os.path.join(path_features, 'features_medium' + tstmp + '.npz')

path_enc = 'encoder_model_' + tstmp + epotag + '.pth'

model_enc.load_state_dict(torch.load(os.path.join(model_path, path_enc), weights_only=True))
model_enc = model_enc.to(device)
_ = model_enc.eval()

train_dataset = SpectroImageDataset(imgpath, edge_attenuation = False)
train_dataset.__len__()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

# extract features (by batches)
feat_li = []
imfiles = []
for i, (data, _, fi) in enumerate(train_loader, 0):    
    print(data.shape)
    data = data.to(device)
    encoded = model_enc(data).detach().cpu().numpy()
    encoded.shape
    feat_li.append(encoded)
    imfiles.append(fi)
    print(len(imfiles))
    # if i > 100:
    #     break

# transform lists to array 
feat = np.concatenate(feat_li)
feat = feat.squeeze()
imfiles = np.concatenate(imfiles)

np.savez(file = path_save, feat=feat, imfiles=imfiles)

# feat.shape
# feat.dtype
# imfiles.shape

