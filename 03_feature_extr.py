#----------------------
#
#
#----------------------

import numpy as np
import pandas as pd
import torch
import os 
from custom_models import Encoder, Decoder, SpectroImageDataset
import pickle

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# imgpath = "C:/xc_real_projects/xc_aec_project_n_europe/downloaded_data_img_24000sps"
imgpath = "C:/xc_real_projects/xc_aec_project_sw_europe/downloaded_data_img_24000sps"

path_models = "C:/xc_real_projects/models"

path_features = "C:/xc_real_projects/features"


tstmp = "20250306_184203"
epotag = '_epo_5'
path_enc = 'encoder_model_' + tstmp + epotag + '.pth'
path_dec = 'decoder_model_' + tstmp + epotag + '.pth'
path_par = 'params_model_'  + tstmp + epotag + '.json'


path_save = os.path.join(path_features, 'features_' + tstmp + '.npz')


with open(os.path.join(path_models, path_par), 'rb') as fp:
    par = pickle.load(fp)


model_enc = Encoder(n_ch_in = par['e']['n_ch_in'], 
                    ch = par['e']['ch'],
                    po = par['e']['po']
                    ) 
model_enc.load_state_dict(torch.load(os.path.join(path_models, path_enc), weights_only=True))
model_enc = model_enc.to(device)
_ = model_enc.eval()

train_dataset = SpectroImageDataset(imgpath)
train_dataset.__len__()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256,  shuffle=True)

# extract features (by batches)
feat_li = []
imfiles = []
for i, (data, _, fi) in enumerate(train_loader, 0):    
    # print(data.shape)
    data = data.to(device)
    encoded = model_enc(data).detach().cpu().numpy()
    # encoded.shape
    feat_li.append(encoded)
    imfiles.append(fi)
    print(len(imfiles))
    if i > 100:
        break

# transform lists to array 
feat = np.concatenate(feat_li)
feat = feat.squeeze()
imfiles = np.concatenate(imfiles)

feat.shape
imfiles.shape

np.savez(file = path_save,  feat=feat, imfiles=imfiles)



