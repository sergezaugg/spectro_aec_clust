#--------------------------------
# Author : Serge Zaugg
# Description : 
#--------------------------------

import os
import json
import plotly.express as px
import torch
from torchsummary import summary
from utils import SpectroImageDataset, make_data_augment_examples, get_models, train_autoencoder
from utils import evaluate_reconstruction_on_examples, encoder_based_feature_extraction, wrap_to_dataset
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#----------------------------------------------------------------------
# (1) train 
with open(os.path.join('./session_params', 'sess_02.json' )) as f:
    sess_info = json.load(f)

train_dataset = SpectroImageDataset(sess_info['imgpath_train'], par = sess_info['data_generator'], augment_1 = True, denoise_1 = False, augment_2 = False, denoise_2 = True)
test_dataset  = SpectroImageDataset(sess_info['imgpath_test'],  par = sess_info['data_generator'], augment_1 = False, denoise_1 = False, augment_2 = False, denoise_2 = True)
make_data_augment_examples(pt_dataset = train_dataset, batch_size = 16).show()
fig02 = make_data_augment_examples(pt_dataset = test_dataset, batch_size = 16).show()
model_enc, model_dec = get_models(sess_info)
summary(model_enc, (1, 128, 1152))
summary(model_dec, (64, 1, 144))
train_autoencoder(sess_info, train_dataset, test_dataset, model_enc, model_dec)

#----------------------------------------------------------------------
# (2) evaluate 
imgpath ="D:/xc_real_projects/example_images/rectangular_1"
n_images = 32
path_trained_models = "D:/xc_real_projects/trained_models"
model_list_B2 = ['20250511_205505','20250512_025001','20250512_215905','20250513_230102',]
model_list_B21 = ['20250514_184030', '20250514_215215']
model_list_B1 = ['20250511_231810','20250512_100933','20250513_001557', '20250514_013412']
model_list_B0 = ['20250512_002141','20250512_124428','20250513_021206',]
model_list_spec = ['20250511_205505','20250514_184030',]
for tstmp in model_list_B21:
    evaluate_reconstruction_on_examples(path_images = imgpath, path_models = path_trained_models, time_stamp_model = tstmp, n_images = 32)

#----------------------------------------------------------------------
# (3) extract 
# path_images = "D:/xc_real_projects/xc_sw_europe/images_24000sps_20250406_092522"
path_images = "D:/xc_real_projects/xc_parus_01/images_24000sps_20250406_081430"
path_models = "D:/xc_real_projects/trained_models"
time_stamp_model = '20250514_184030'

di = encoder_based_feature_extraction(path_images, path_models, time_stamp_model, devel = False)
di['feature_array'].shape
di['image_file_name_array'].shape
wrap_to_dataset(di)












