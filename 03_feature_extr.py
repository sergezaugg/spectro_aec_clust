#----------------------
# Author : Serge Zaugg
# Description : 
#----------------------

import numpy as np
import pandas as pd
import os 
import pickle
import shutil
# from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import datetime
import torch
from custom_models import SpectroImageDataset, EncoderAvgpool
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define paths
model_path = "C:/xc_real_projects/models/encoder_model_20250319_210308_epo_20.pth"

path_xc = "C:/xc_real_projects/xc_sw_europe/"

imgpath = os.path.join(path_xc, "images_24000sps_20250406_092522")
meta_path = os.path.join(path_xc, "downloaded_data_meta.pkl")

# load metadata 
df_meta = pd.read_pickle(meta_path)
# df_meta.head()
# df_meta.shape

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
    if i > 6:
        break

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
path_save_npz = os.path.join(features_save_path, 'features_from_encoder' + tstmp + '.pkl')
encoder_id = np.array(model_path)

dat_di = {
    'feat': feat,
    'imfiles': imfiles,
     'encoder_id' : encoder_id,
    'meta': df_meta
    }

with open(path_save_npz, 'wb') as handle:
    pickle.dump(dat_di, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)

shutil.make_archive(os.path.join(features_save_path, 'images'), 'zip', imgpath)



feat.shape
imfiles.shape
encoder_id.shape
df_meta.shape



















