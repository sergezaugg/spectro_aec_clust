#----------------------
# Author : Serge Zaugg
# Description : 
#----------------------

import plotly.express as px
import torch
import numpy as np
from custom_models import SpectroImageDataset
from plotly.subplots import make_subplots

# imgpath_train = "C:/xc_real_projects/da_examples/1"
imgpath_train = "C:/xc_real_projects/da_examples/2"
# imgpath_train = "C:/xc_real_projects/da_examples/3"

# define data loader 
train_dataset = SpectroImageDataset(imgpath_train, edge_attenuation = True)
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
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(px.imshow(img_orig).data[0], row=1, col=1)
    fig.add_trace(px.imshow(img_augm).data[0], row=1, col=2)
    _ = fig.update_layout(autosize=True,height=550, width = 1000)
    fig.show()





