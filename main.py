#--------------------------------
# Author : Serge Zaugg
# Description : 
#--------------------------------


# import numpy as np
# import torch
# import plotly.express as px
# import os 
# from utils import SpectroImageDataset, make_data_augment_examples
# from plotly.subplots import make_subplots
from mlutils import evaluate_reconstruction_on_examples, encoder_based_feature_extraction, wrap_to_dataset


# (1) train 




# (2) evaluate 
imgpath ="D:/xc_real_projects/example_images/rectangular_1"
n_images = 32
path_trained_models = "D:/xc_real_projects/trained_models"

model_list_B2 = ['20250511_205505','20250512_025001','20250512_215905','20250513_230102',]
model_list_B1 = ['20250511_231810','20250512_100933','20250513_001557', '20250514_013412']
model_list_B0 = ['20250512_002141','20250512_124428','20250513_021206',]

for tstmp in model_list_B2:
    evaluate_reconstruction_on_examples(path_images = imgpath, path_models = path_trained_models, time_stamp_model = tstmp, n_images = 32,)




# (3) extract 
path_images = "D:/xc_real_projects/xc_sw_europe/images_24000sps_20250406_092522"
path_models = "D:/xc_real_projects/trained_models"
time_stamp_model = '20250512_215905'

di = encoder_based_feature_extraction(path_images, path_models, time_stamp_model, devel = True)

di['feature_array'].shape
di['image_file_name_array'].shape
di['path_images']
di['path_encoder']

wrap_to_dataset(di)





