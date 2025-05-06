#----------------------
# Author : Serge Zaugg
# Description : 
#----------------------

import plotly.express as px
import torch
import numpy as np
from utils import SpectroImageDataset
from plotly.subplots import make_subplots

# imgpath_train = "D:/xc_real_projects/da_examples/1"
imgpath_train = "D:/xc_real_projects/xc_ne_europe/images_24000sps_20250406_095331"




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
       'thld' :   0.25, 
        } 
    }


# par = {
#     'da': {
#         'rot_deg' : 0.0,
#         'trans_prop' : 0.00,
#         'brightness' : 0.0, 
#         'contrast'   : 0.0,
#         'gnoisesigm' : 0.0,
#         'gnoiseprob' : 0.0,
#         },
#     'den': {  
#        'thld' :   0.25, 
#         } 
#     }


# define TRAIN data loader 
train_dataset = SpectroImageDataset(imgpath_train, par = par , augment_1 = True, denoise_1 = False,    augment_2 = True, denoise_2 = True)


# # define TEST data loader 
# train_dataset = SpectroImageDataset(imgpath_train, par = par , augment_1 = True, denoise_1 = False,    augment_2 = False, denoise_2 = True)




train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12,  shuffle=True, drop_last=True)
for i, (da_1, da_2, fi) in enumerate(train_loader, 0):
    if i > 0:
        break
    print(da_1.shape)
    print(da_2.shape)

print(da_1.min(), da_1.max(), da_2.min(), da_2.max())


# assess data augmentation 
for ii in np.random.randint(da_2.shape[0], size = 12):
    print(ii)
    img_1 = da_1[ii].cpu().detach().numpy()
    img_1 = img_1.squeeze() 
    img_1 = 255*img_1 #(img_1 - img_1.min())/(img_1.max())

    img_2 = da_2[ii].cpu().detach().numpy()
    img_2 = img_2.squeeze() 
    img_2 = 255*img_2 #(img_2 - img_2.min())/(img_2.max())

    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(px.imshow(img_1).data[0], row=1, col=1)
    fig.add_trace(px.imshow(img_2).data[0], row=2, col=1)
    fig.show()





