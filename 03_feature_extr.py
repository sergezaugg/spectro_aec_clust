#----------------------
#
#
#----------------------

import numpy as np
import pandas as pd
import torch
import os 
from custom_models import SpectroImageDataset, EncoderAvgpool, DecoderTransp, EncoderNopad
import pickle

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

imgpath = "C:/xc_real_projects/xc_aec_project_sw_europe/downloaded_data_img_24000sps"

path_features = "C:/xc_real_projects/features"

model_path = "C:/xc_real_projects/models"


# tstmp = "20250308_214623" #   good
# epotag = '_epo_28'
# model_enc = EncoderAvgpool()
# model_dec = DecoderTransp()

tstmp = '20250309_005037' # not so good
epotag = '_epo_30'
model_enc = EncoderNopad()
model_dec = DecoderTransp()


path_save = os.path.join(path_features, 'features_' + tstmp + '.npz')

path_enc = 'encoder_model_' + tstmp + epotag + '.pth'
path_dec = 'decoder_model_' + tstmp + epotag + '.pth'


model_enc.load_state_dict(torch.load(os.path.join(model_path, path_enc), weights_only=True))
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
    # if i > 100:
    #     break

# transform lists to array 
feat = np.concatenate(feat_li)
feat = feat.squeeze()
imfiles = np.concatenate(imfiles)

feat.shape
imfiles.shape

np.savez(file = path_save,  feat=feat, imfiles=imfiles)



