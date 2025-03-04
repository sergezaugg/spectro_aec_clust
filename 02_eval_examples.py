#--------------------------------
# Assess trained models by direct comparison of a few reconstructed images
#--------------------------------

import numpy as np
import pandas as pd
import torch
import plotly.express as px
import os 
from PIL import Image
from custom_models import Encoder, Decoder, SpectroImageDataset
import pickle
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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


model_dec = Decoder(n_ch_out = par['d']['n_ch_out'], 
                    ch = par['d']['ch'], 
                    po = par['d']['po'], 
                    )
model_dec.load_state_dict(torch.load(os.path.join(model_path, path_dec), weights_only=True))
model_dec = model_dec.to(device)
_ = model_dec.eval()

# _ = model_enc.eval()
# _ = model_dec.eval()

# test 
test_dataset = SpectroImageDataset(imgpath)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=128,  shuffle=True, drop_last=False)

# test loss 
for i_test, (da_orig, data_augm, fi) in enumerate(test_loader, 0):
    print(i_test)
    if i_test > 0:
        break

_ = model_enc.eval()
_ = model_dec.eval()

data = da_orig.to(device)
encoded = model_enc(data).to(device)
decoded = model_dec(encoded).to(device)

# ii = 489 
for ii in np.random.randint(data.shape[0], size = 15):
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



