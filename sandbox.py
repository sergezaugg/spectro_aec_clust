#----------------------
# Author : Serge Zaugg
# Description : 
#----------------------

import plotly.express as px
import torch
import numpy as np
from utils import SpectroImageDataset, make_data_augment_examples
from utils import SpectroImageDataset 

imgpath_train = "D:/xc_real_projects/da_examples/1"
# imgpath_train = "D:/xc_real_projects/xc_ne_europe/images_24000sps_20250406_095331"

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

# define TRAIN data loader 
train_dataset = SpectroImageDataset(imgpath_train, par = par , augment_1 = True, denoise_1 = False,    augment_2 = True, denoise_2 = True)

fig01 = make_data_augment_examples(pt_dataset = train_dataset)
fig01.show()





