#--------------------------------
# Author : Serge Zaugg
# Description : Assess trained models by direct comparison of a few reconstructed images
#--------------------------------

import numpy as np
import torch
import plotly.express as px
import os 
from custom_models_2 import EncoderAvgpool, DecoderTranspNew
from utils import SpectroImageDataset, make_data_augment_examples
from plotly.subplots import make_subplots
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# set paths 
imgpath = "D:/xc_real_projects/xc_corvidae_01/images_24000sps_20250415_181912"
model_path = "D:/xc_real_projects/models"

# tstmp = '20250507_153715'
# tstmp = '20250419_033651'
tstmp = '20250418_182756'







path_enc = [a for a in os.listdir(model_path) if tstmp in a and 'encoder_model_' in a][0]
path_dec = [a for a in os.listdir(model_path) if tstmp in a and 'decoder_model_' in a][0]

# path_enc = 'encoder_model_20250507_125957_epo_9.pth' 
# path_dec = 'decoder_model_20250507_125957_epo_9.pth'

n_images = 16


# load trained AEC
model_enc = EncoderAvgpool()
model_dec = DecoderTranspNew()

model_enc.load_state_dict(torch.load(os.path.join(model_path, path_enc), weights_only=True))
model_enc = model_enc.to(device)
_ = model_enc.eval()

model_dec.load_state_dict(torch.load(os.path.join(model_path, path_dec), weights_only=True))
model_dec = model_dec.to(device)
_ = model_dec.eval()

# get a few images in array format 
test_dataset = SpectroImageDataset(imgpath, par = None, augment_1 = False, denoise_1 = False, augment_2 = False, denoise_2 = False)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=n_images,  shuffle = False, drop_last = False)



# fig02 = make_data_augment_examples(pt_dataset = test_dataset, batch_size = 16)
# fig02.show()




for i_test, (data_1, _ , _ ) in enumerate(test_loader, 0):
    if i_test > 0: break
    print(data_1.shape)
  
# predict 
data = data_1.to(device)
encoded = model_enc(data).to(device)
decoded = model_dec(encoded).to(device)

# plot 
fig = make_subplots(rows=n_images, cols=2,)
for ii in range(n_images) : 
    img_orig = data[ii].cpu().numpy()
    img_orig = img_orig.squeeze() # 1 ch
    img_orig = 255.0*img_orig   
    img_reco = decoded[ii].cpu().detach().numpy()
    img_reco = img_reco.squeeze()  # 1 ch
    img_reco = 255.0*img_reco   
    fig.add_trace(px.imshow(img_orig).data[0], row=ii+1, col=1)
    fig.add_trace(px.imshow(img_reco).data[0], row=ii+1, col=2)
    fig.update_layout(autosize=True,height=400*n_images, width = 800)
fig.show()



