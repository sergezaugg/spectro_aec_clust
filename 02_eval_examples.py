#--------------------------------
# Assess trained models by direct comparison of a few reconstructed images
#--------------------------------

import os
import plotly.express as px
import torch
import numpy as np
# from custom_utils import CifarDataset
# from custom_models import Encoder, Decoder
from plotly.subplots import make_subplots
import yaml
import pandas as pd
from ptutils import SpectroImageDataset, Encoder, Decoder


model_path = "C:/xc_real_projects/models"

<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# with open('./config.yaml') as f:
#     conf = yaml.safe_load(f)

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -----------------------------------
# (0) set script parameters
n_img = 10
rand_seed = 4589
select_latent_size = 128 # 1024
min_n_epochs =  99


# -----------------------------------
# (1) load a subset of models, store in a dict and sort the dict by nb of conv blocks
all_models = [a for a in os.listdir(model_path) if '.pt' in a]
models_di = {}
for m_i , pt_model in enumerate(all_models):
    print(m_i , pt_model)
    trained_model_object = torch.load(os.path.join(model_path, pt_model), weights_only = False)
    # trained_model_object.keys()
    par = trained_model_object['train_session_params']
    # print(par['n_epochs'])
    if par['n_epochs'] < min_n_epochs:
        continue
    # print(m_i , pt_model)
    if par['enc']['n_ch_latent'] != select_latent_size:
        continue
    # make unique keys  
    extended_key = str(par['enc']['n_conv_blocks']) + "_" + str(m_i)
    models_di[extended_key] = trained_model_object
models_di = dict(sorted(models_di.items()))
# models_di.keys()
n_models = len(models_di)


# -----------------------------------
# (2) loop over models and a few images to plot original and reconstructed side-by-side
lat_siz_tag = "Latent " + "<br>" + str(select_latent_size) + "<br>"
col_tit = ["original"] + [lat_siz_tag  + 'n_conv ' + a[0] for a in models_di]
fig = make_subplots(rows=n_img, cols=n_models+1+2, 
    shared_yaxes=True, horizontal_spacing = 0.02, vertical_spacing = 0.01,  
    column_titles = col_tit
    )
for counter, m_i in enumerate(models_di):    
    print(counter, m_i)

    trained_model_object = models_di[m_i]
    # trained_model_object.keys()
    par = trained_model_object['train_session_params']

    # initialize models  
    model_enc = Encoder(
        n_ch_in         = par['enc']['n_ch_in'], 
        n_ch_latent     = par['enc']['n_ch_latent'], 
        shape_input     = par['enc']['shape_input'], 
        n_conv_blocks   = par['enc']['n_conv_blocks'],
        ch              = par['enc']['n_ch_convs'], 
        po              = par['enc']['post_conv_pool'], 
        )
    
    model_dec = Decoder(
        n_ch_out        = par['dec']['n_ch_out'], 
        n_ch_latent     = par['dec']['n_ch_latent'], 
        shape_output    = par['dec']['shape_output'], 
        n_conv_blocks   = par['dec']['n_conv_blocks'],
        ch              = par['dec']['n_ch_convs'], 
        po              = par['dec']['post_conv_pool'], 
        )
   
    model_enc.load_state_dict(trained_model_object['encoder_weights'])
    model_enc = model_enc.to(device)
    _ = model_enc.eval()

    model_dec.load_state_dict(trained_model_object['decoder_weights'])
    model_dec = model_dec.to(device)
    _ = model_dec.eval()

    # get a single small batch
    test_dataset = CifarDataset(path_to_cifar_data = conf['path_data_test'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = n_img, shuffle = True)
    torch.manual_seed(rand_seed) # to get same batch every time
    i, data = next(enumerate(test_loader))
    # data.shape

    # check a few examples 
    data = data.to(device)
    encoded = model_enc(data).to(device)
    decoded = model_dec(encoded).to(device)

    # throw all this to an image 
    for ii in range(n_img):
        img_orig = data[ii].cpu().numpy()
        img_orig = img_orig.transpose(1,2,0)
        img_orig = 255*(img_orig - img_orig.min())/(img_orig.max())

        img_reco = decoded[ii].cpu().detach().numpy()
        img_reco = img_reco.transpose(1,2,0)
        img_reco = 255*(img_reco - img_reco.min())/(img_reco.max())

        _ = fig.add_trace(px.imshow(img_orig, aspect ='equal').data[0], row=ii+1, col=1)
        _ = fig.add_trace(px.imshow(img_reco, aspect ='equal').data[0], row=ii+1, col=counter+2)

_ = fig.update_layout(width=1500)
_ = fig.update_xaxes(showticklabels=False) 
_ = fig.update_yaxes(showticklabels=False) 

fig.show()

        


