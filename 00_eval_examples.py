#--------------------------------
# Assess trained models by direct comparison of a few reconstructed images
#--------------------------------

import numpy as np
import pandas as pd
import torch
import plotly.express as px
import os 
from custom_models_2 import EncoderAvgpool, DecoderTranspNew
from utils import SpectroImageDataset
from plotly.subplots import make_subplots
import plotly.graph_objects as go

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

imgpath = "D:/xc_real_projects/xc_corvidae_01/images_24000sps_20250415_181912"

model_path = "D:/xc_real_projects/models"

# tstmp = '20250315_132629'
# epotag = '_epo_20'
tstmp = '20250419_033651'
epotag = '_epo_20'
model_enc = EncoderAvgpool()
model_dec = DecoderTranspNew()


path_enc = 'encoder_model_' + tstmp + epotag + '.pth'
path_dec = 'decoder_model_' + tstmp + epotag + '.pth'

model_enc.load_state_dict(torch.load(os.path.join(model_path, path_enc), weights_only=True))
model_enc = model_enc.to(device)
_ = model_enc.eval()

model_dec.load_state_dict(torch.load(os.path.join(model_path, path_dec), weights_only=True))
model_dec = model_dec.to(device)
_ = model_dec.eval()

# test 
test_dataset = SpectroImageDataset(imgpath, edge_attenuation = False, do_augment = False)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=128,  shuffle=True, drop_last=False)

# test loss 
for i_test, (da_orig, data_augm, fi) in enumerate(test_loader, 0):
    print(i_test)
    print(da_orig.shape)
    if i_test > 1:
        break

_ = model_enc.eval()
_ = model_dec.eval()

data = da_orig.to(device)
encoded = model_enc(data).to(device)
decoded = model_dec(encoded).to(device)

# ii = 489 
np.random.seed(45623)
for ii in np.random.randint(data.shape[0], size = 15):
# for ii in [0,1,2]:
    img_orig = data[ii].cpu().numpy()
    img_orig = img_orig.squeeze() # 1 ch
    img_orig = 255*(img_orig - img_orig.min())/(img_orig.max())
    # fig00 = px.imshow(img_orig, height = 500, title="original")
    img_reco = decoded[ii].cpu().detach().numpy()
    img_reco = img_reco.squeeze()  # 1 ch
    img_reco = 255*(img_reco - img_reco.min())/(img_reco.max())
    # fig01 = px.imshow(img_reco, height = 500, title="reconstructed")
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(px.imshow(img_orig).data[0], row=1, col=1)
    fig.add_trace(px.imshow(img_reco).data[0], row=1, col=2)
    fig.update_layout(autosize=True,height=550, width = 1000)
    fig.show()



