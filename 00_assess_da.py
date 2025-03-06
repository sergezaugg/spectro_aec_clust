#----------------------
# Author : Serge Zaugg
# Description : 
#----------------------

import os 
import plotly.express as px
import torch
import numpy as np
from custom_models import SpectroImageDataset
from plotly.subplots import make_subplots
from torchvision.transforms import v2

# imgpath_train = "C:/xc_real_projects/da_examples/1"
imgpath_train = "C:/xc_real_projects/da_examples/2"
# imgpath_train = "C:/xc_real_projects/da_examples/3"

#----------------------
# define data loader 
train_dataset = SpectroImageDataset(imgpath_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16,  shuffle=True, drop_last=True)
for i, (da_orig, data, fi) in enumerate(train_loader, 0):
    if i > 0:
        break
    print(data.shape)
    print(da_orig.shape)

# assess data augmentation 
for ii in np.random.randint(data.shape[0], size = 16):
    print(ii)
    img_orig = da_orig[ii].cpu().detach().numpy()
    img_orig = img_orig.squeeze() 
    img_orig = 255*(img_orig - img_orig.min())/(img_orig.max())

    img_augm = data[ii].cpu().detach().numpy()
    img_augm = img_augm.squeeze() 
    img_augm = 255*(img_augm - img_augm.min())/(img_augm.max())

    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(px.imshow(img_orig).data[0], row=1, col=1)
    fig.add_trace(px.imshow(img_augm).data[0], row=1, col=2)
    _ = fig.update_layout(autosize=True,height=550, width = 1000)
    fig.show()




# from PIL import Image

# pa = "C:/xc_real_projects/xc_aec_project_n_europe/downloaded_data_img_24000sps/Corvus_corax_XC127693_Raven_call_flyby_B06h43m16s25may2012_segm_16.0.png"
# img = Image.open( pa)
# img_arr = np.asarray(img).copy()
# thld = np.quantile(img_arr, q=0.75)
# img_arr[img_arr < thld] = 0.0
# im =Image.fromarray(img_arr)



# # simple denoising with threshold 
# img_arr = np.asarray(img).copy()
# thld = np.quantile(img_arr, q=0.95)
# img_arr[img_arr < thld] = 0.0
# img =Image.fromarray(img_arr)


