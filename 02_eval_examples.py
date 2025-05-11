#--------------------------------
# Author : Serge Zaugg
# Description : Assess trained models by direct comparison of a few reconstructed images
#--------------------------------

import numpy as np
import torch
import plotly.express as px
import os 
from utils import SpectroImageDataset, make_data_augment_examples
from plotly.subplots import make_subplots
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




# ---------------------
# (1) load a few images 
# set paths 
imgpath = "D:/xc_real_projects/xc_corvidae_01/images_24000sps_20250415_181912"

n_images = 32


# default 1 
par = {
    'da': {
        'rot_deg'    : 0.30,  # ok
        'trans_prop' : 0.005, # ok
        'brightness' : 0.40,  # ok
        'contrast'   : 0.40,  # ok
        'gnoisesigm' : 0.10,  # ok
        'gnoiseprob' : 0.50,  # ok
        },
    'den': {  
       'thld' :   0.50, 
        } 
    }


# get a few images in array format 
test_dataset = SpectroImageDataset(imgpath, par = par, augment_1 = False, denoise_1 = False, augment_2 = False, denoise_2 = True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=n_images,  shuffle = True, drop_last = False)

# fig02 = make_data_augment_examples(pt_dataset = test_dataset, batch_size = 16)
# fig02.show()

for i_test, (data_1, data_2 , _ ) in enumerate(test_loader, 0):
    if i_test > 0: break
    print(data_1.shape)
    print(data_2.shape)

# ---------------------




# ---------------------
# (2) predict 
path_trained_models = "D:/xc_real_projects/trained_models"


# tstmp = '20250510_193201'
# tstmp = '20250511_005328'
tstmp = '20250511_032255'




path_enc = [a for a in os.listdir(path_trained_models) if tstmp in a and 'encoder_model_' in a][0]
path_dec = [a for a in os.listdir(path_trained_models) if tstmp in a and 'decoder_model_' in a][0]

# load trained AEC
model_enc = torch.load( os.path.join(path_trained_models, path_enc),  weights_only = False)
model_dec = torch.load( os.path.join(path_trained_models, path_dec),  weights_only = False)

model_enc = model_enc.to(device)
_ = model_enc.eval()

model_dec = model_dec.to(device)
_ = model_dec.eval()




# predict 
data = data_1.to(device)
encoded = model_enc(data).to(device)
decoded = model_dec(encoded).to(device)

# plot 
fig = make_subplots(rows=n_images, cols=2,)
for ii in range(n_images) : 

    img_orig = data_2[ii].cpu().numpy()
    img_orig = img_orig.squeeze() # 1 ch
    img_orig = 255.0*img_orig  

    img_reco = decoded[ii].cpu().detach().numpy()
    img_reco = img_reco.squeeze()  # 1 ch
    img_reco = 255.0*img_reco   
    _ = fig.add_trace(px.imshow(img_orig).data[0], row=ii+1, col=1)
    _ = fig.add_trace(px.imshow(img_reco).data[0], row=ii+1, col=2)
_ = fig.update_layout(autosize=True,height=400*n_images, width = 800)
_ = fig.update_layout(title="Model ID: " + tstmp)
fig.show()
# ---------------------



