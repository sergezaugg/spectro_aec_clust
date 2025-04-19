#----------------------
# Author : Serge Zaugg
# Description : 
#----------------------

import plotly.express as px
import torch
import numpy as np
from utils import SpectroImageDataset
from plotly.subplots import make_subplots

# imgpath_train = "D:/xc_real_projects/da_examples/4"
imgpath_train = "D:/xc_real_projects/xc_ne_europe/images_24000sps_20250406_095331"

# define data loader 
train_dataset = SpectroImageDataset(imgpath_train, augment_1=True, augment_2=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16,  shuffle=True, drop_last=True)
for i, (da_orig, data, fi) in enumerate(train_loader, 0):
    if i > 0:
        break
    print(da_orig.shape)
    print(data.shape)

# assess data augmentation 
for ii in np.random.randint(data.shape[0], size = 16):
    print(ii)
    img_orig = da_orig[ii].cpu().detach().numpy()
    img_orig = img_orig.squeeze() 
    img_orig = 255*(img_orig - img_orig.min())/(img_orig.max())
    img_augm = data[ii].cpu().detach().numpy()
    img_augm = img_augm.squeeze() 
    img_augm = 255*(img_augm - img_augm.min())/(img_augm.max())
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(px.imshow(img_orig).data[0], row=1, col=1)
    fig.add_trace(px.imshow(img_augm).data[0], row=2, col=1)
    # _ = fig.update_layout(autosize=True,height=200, width = 1000)
    fig.show()





