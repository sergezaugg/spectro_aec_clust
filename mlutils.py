#--------------------------------
# Author : Serge Zaugg
# Description : bigger ML processes are wrapped into functions here
#--------------------------------

import numpy as np
import torch
import plotly.express as px
import os 
from utils import SpectroImageDataset
from plotly.subplots import make_subplots
import pickle
import pandas as pd
import shutil
import datetime
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def evaluate_reconstruction_on_examples(
        path_images,
        path_models,
        time_stamp_model,
        n_images = 32,
        ):
    """
    Assess trained models by direct comparison of a few reconstructed images
    """

    # ---------------------
    # (1) load a few images 

    # lod info from training session 
    path_sess = [a for a in os.listdir(path_models) if time_stamp_model in a and '_session_info' in a][0]
    with open(os.path.join(path_models, path_sess), 'rb') as f:
        di_sess = pickle.load(f)

    # load data generator params used during training 
    par = di_sess['sess_info']['data_generator']

    # get a few images in array format 
    test_dataset = SpectroImageDataset(path_images, par = par, augment_1 = False, denoise_1 = False, augment_2 = False, denoise_2 = True)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=n_images,  shuffle = False, drop_last = False)

    for i_test, (data_1, data_2 , _ ) in enumerate(test_loader, 0):
        if i_test > 0: break
        print(data_1.shape)
        print(data_2.shape)

    # ---------------------
    # (2) load models  
    path_enc = [a for a in os.listdir(path_models) if time_stamp_model in a and 'encoder_model' in a][0]
    path_dec = [a for a in os.listdir(path_models) if time_stamp_model in a and 'decoder_model' in a][0]
    # load trained AEC
    model_enc = torch.load( os.path.join(path_models, path_enc), weights_only = False)
    model_dec = torch.load( os.path.join(path_models, path_dec), weights_only = False)
    model_enc = model_enc.to(device)
    model_dec = model_dec.to(device)
    _ = model_enc.eval()
    _ = model_dec.eval()

    # ---------------------
    # (3) predict 
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
    _ = fig.update_layout(title="Model ID: " + time_stamp_model)
    fig.show()
    # ---------------------


def encoder_based_feature_extraction(
    path_images,
    path_models ,
    time_stamp_model ,
    batch_size = 128,
    shuffle = True,
    devel = False,
    ):
    """
    Description:
    Arguments:
    """
    # get the file corresponding to the time stamp
    path_enc = [a for a in os.listdir(path_models) if time_stamp_model in a and 'encoder_model' in a][0]
    # load trained AEC
    model_enc = torch.load( os.path.join(path_models, path_enc),  weights_only = False)
    model_enc = model_enc.to(device)
    _ = model_enc.eval()
    # prepare dataloader ()
    test_dataset = SpectroImageDataset(path_images, augment_1 = False, denoise_1 = False, augment_2 = False, denoise_2 = False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = shuffle)
    # extract features (by batches)
    feat_li = []
    imfiles = []
    for i, (data, _, fi) in enumerate(test_loader, 0):    
        print(data.shape)
        data = data.to(device)
        encoded = model_enc(data).detach().cpu().numpy()
        encoded.shape
        feat_li.append(encoded)
        imfiles.append(fi)
        print(len(imfiles))
        if devel:
            if i > 1:
                break
    # transform lists to array 
    feat = np.concatenate(feat_li)
    feat = feat.squeeze()
    imfiles = np.concatenate(imfiles)
    return(feat, imfiles)


def wrap_to_dataset(feat, imfiles, path_xc_dir, image_dir):
    # load metadata 
    meta_path = os.path.join(path_xc_dir, "downloaded_data_meta.pkl")
    df_meta = pd.read_pickle(meta_path)
    # save all relevant objects 
    tstmp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
    features_save_path = os.path.join(path_xc_dir, "features" + tstmp) 
    if not os.path.exists(features_save_path):
        os.makedirs(features_save_path)
    path_save_npz = os.path.join(features_save_path, 'features_from_encoder' + tstmp + '.pkl')
    # encoder_id = np.array(os.path.join(path_models, path_enc))
    dat_di = {
        'feat': feat,
        'imfiles': imfiles,
        # 'encoder_id' : encoder_id,
        'meta': df_meta
        }
    with open(path_save_npz, 'wb') as handle:
        pickle.dump(dat_di, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # make zip
    path_images = os.path.join(path_xc_dir, image_dir)
    shutil.make_archive(os.path.join(features_save_path, 'images'), 'zip', path_images)
    shutil.move(os.path.join(features_save_path, 'images.zip'), os.path.join(features_save_path, 'images.speczip'))

      





# devel 
if __name__ == "__main__":

    # # imgpath ="D:/xc_real_projects/example_images/rectangular_1"
    # # n_images = 32
    # # path_trained_models = "D:/xc_real_projects/trained_models"
    # # tstmp = '20250512_025001'
    # evaluate_reconstruction_on_examples(
    #         path_images ="D:/xc_real_projects/example_images/rectangular_1",
    #         path_models = "D:/xc_real_projects/trained_models",
    #         time_stamp_model = '20250512_025001',
    #         n_images = 32,
    #         )
    

        
    path_images = "D:/xc_real_projects/xc_sw_europe/images_24000sps_20250406_092522"
    path_models = "D:/xc_real_projects/trained_models"
    time_stamp_model = '20250511_231810'

    feat, imfiles = encoder_based_feature_extraction(path_images, path_models, time_stamp_model, devel = True)

    feat.shape
    imfiles.shape





    path_xc_dir = "D:/xc_real_projects/xc_sw_europe/"
    image_dir = "images_24000sps_20250406_092522"

    wrap_to_dataset(feat, imfiles, path_xc_dir, image_dir)



