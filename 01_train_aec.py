#----------------------
# Author : Serge Zaugg
# Description : 
#----------------------

import plotly.express as px
import torch
from torchsummary import summary
from utils import SpectroImageDataset, make_data_augment_examples
from mlutils import get_models, train_autoencoder
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


path_untrained_models = "D:/xc_real_projects/untrained_models"

path_trained_models = "D:/xc_real_projects/trained_models"

sess_info = {
    'imgpath_train' : "D:/xc_real_projects/xc_ne_europe/images_24000sps_20250406_095331",
    'imgpath_test'  : "D:/xc_real_projects/xc_sw_europe/images_24000sps_20250406_092522",
    'batch_size_tr' : 8,  # 8,
    'batch_size_te' : 32,
    'n_epochs' : 10,

    # 'hot_start' : False, 
    # 'model_tag' : 'gen_B0',  

    'hot_start' : True, 
    'model_tag' : '20250513_001557',  

    # 
    'data_generator' : {
        'da': {
            'rot_deg'    : 0.30,  # ok
            'trans_prop' : 0.005, # ok
            'brightness' : 0.40,  # ok
            'contrast'   : 0.40,  # ok
            'gnoisesigm' : 0.10,  # ok
            'gnoiseprob' : 0.50,  # ok
            },
        'den': {  
        'thld' :  0.40, #  0.30, 
            } 
        }
    }













#----------------------
# define data loader 
train_dataset = SpectroImageDataset(sess_info['imgpath_train'], par = sess_info['data_generator'], augment_1 = True, denoise_1 = False, augment_2 = False, denoise_2 = True)
test_dataset  = SpectroImageDataset(sess_info['imgpath_test'],  par = sess_info['data_generator'], augment_1 = False, denoise_1 = False, augment_2 = False, denoise_2 = True)



# get some info 
train_dataset.__len__()
test_dataset.__len__()
tra_itm = train_dataset.__getitem__(45)
tes_itm = test_dataset.__getitem__(45)
tra_itm[0].shape
tra_itm[1].shape
tes_itm[0].shape
tes_itm[1].shape



fig01 = make_data_augment_examples(pt_dataset = train_dataset, batch_size = 16)
fig01.show()

fig02 = make_data_augment_examples(pt_dataset = test_dataset, batch_size = 16)
fig02.show()


model_enc, model_dec = get_models(sess_info, path_untrained_models, path_trained_models)


model_enc = model_enc.to(device)
summary(model_enc, (1, 128, 1152))

model_dec = model_dec.to(device)
summary(model_dec, (128, 1, 72))
summary(model_dec, (128, 1, 144))

train_autoencoder(sess_info, train_dataset, test_dataset, model_enc, model_dec, path_trained_models)

