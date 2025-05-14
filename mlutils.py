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
import torch.nn as nn
import torch.optim as optim
from utils import SpectroImageDataset
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# load  models 
def get_models(sess_info, path_untrained_models, path_trained_models):
    if sess_info['hot_start'] == False:
        tstmp_0 = sess_info['model_tag']
        path_enc = [a for a in os.listdir(path_untrained_models) if tstmp_0 in a and 'cold_encoder' in a][0]
        path_dec = [a for a in os.listdir(path_untrained_models) if tstmp_0 in a and 'cold_decoder' in a][0]
        model_enc = torch.load(os.path.join(path_untrained_models, path_enc), weights_only = False)
        model_dec = torch.load(os.path.join(path_untrained_models, path_dec), weights_only = False)
        sess_info['model_gen'] = sess_info['model_tag']
    elif sess_info['hot_start'] == True:
        tstmp_1 = sess_info['model_tag']
        path_enc = [a for a in os.listdir(path_trained_models) if tstmp_1 in a and 'encoder_model' in a][0]
        path_dec = [a for a in os.listdir(path_trained_models) if tstmp_1 in a and 'decoder_model' in a][0]
        model_enc = torch.load(os.path.join(path_trained_models, path_enc), weights_only = False)
        model_dec = torch.load(os.path.join(path_trained_models, path_dec), weights_only = False) 
        # load info from previous training session 
        path_sess = [a for a in os.listdir(path_trained_models) if tstmp_1 in a and '_session_info' in a][0]
        with open(os.path.join(path_trained_models, path_sess), 'rb') as f:
            di_origin_sess = pickle.load(f)
        # load model generation 
        sess_info['model_gen'] = di_origin_sess['sess_info']['model_gen']
    else:
        print("something is wrong with sess_info['hot_start']")
    return(model_enc, model_dec)


def train_autoencoder(sess_info, train_dataset, test_dataset, model_enc, model_dec, path_trained_models):

    # train 
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=sess_info['batch_size_tr'],  shuffle=True, drop_last=True)
    test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size=sess_info['batch_size_te'],  shuffle=True, drop_last=True)

    # instantiate loss, optimizer
    criterion = nn.MSELoss() #nn.BCELoss()
    optimizer = optim.Adam(list(model_enc.parameters()) + list(model_dec.parameters()), lr=0.001)
    # optimizer = optim.SGD(list(model_enc.parameters()) + list(model_dec.parameters()), lr=0.01, momentum=0.9)

    n_batches_tr = train_dataset.__len__() // sess_info['batch_size_tr']
    n_batches_te = test_dataset.__len__() // sess_info['batch_size_te']

    mse_test_li = []
    mse_trai_li = []
    for epoch in range(sess_info['n_epochs']):
        print(f"Epoch: {epoch + 1}/{sess_info['n_epochs']}")
        #----------------
        # Train the model 
        _ = model_enc.train()
        _ = model_dec.train()
        trai_perf_li = []
        for batch_tr, (da_tr_1, da_tr_2, fi) in enumerate(train_loader, 0):
            # if True:
            #     if batch_tr > 10: break
            da_tr_1 = da_tr_1.to(device)
            da_tr_2 = da_tr_2.to(device)
            # reset the gradients 
            optimizer.zero_grad()
            # forward 
            encoded = model_enc(da_tr_1)
            # encoded.shape
            decoded = model_dec(encoded)
            # compute the reconstruction loss 
            loss = criterion(decoded, da_tr_2)
            trai_perf_li.append(loss.cpu().detach().numpy().item())
            # compute the gradients
            loss.backward()
            # update the weights
            optimizer.step()
            # feedback every 10th batch
            if batch_tr % 10 == 0:
                print('loss', np.round(loss.item(),5), " --- "  + str(batch_tr) + " out of " + str(n_batches_tr) + " batches")
                print(decoded.cpu().detach().numpy().min().round(3) , decoded.cpu().detach().numpy().max().round(3) )
                print("-")
        mse_trai_li.append(np.array(trai_perf_li).mean())        

        #----------------------------------
        # Testing the model at end of epoch 
        _ = model_enc.eval()
        _ = model_dec.eval()
        with torch.no_grad():
            test_perf_li = []
            for btchi, (da_te_1, da_te_2, fi) in enumerate(test_loader, 0):
                if btchi > 10: break # 100
                da_te_1 = da_te_1.to(device)
                da_te_2 = da_te_2.to(device)
                # forward 
                encoded = model_enc(da_te_1)#.to(device)
                # encoded.shape
                decoded = model_dec(encoded)#.to(device)
                # compute the reconstruction loss 
                loss_test = criterion(decoded, da_te_2)
                test_perf_li.append(loss_test.cpu().detach().numpy().item())
                # feedback every 10th batch
                if btchi % 10 == 0:
                    print('TEST loss', np.round(loss_test.item(),5), " --- "  + str(btchi) + " out of " + str(n_batches_te) + " batches")
            mse_test_li.append(np.array(test_perf_li).mean())
        #----------------------------------
        
    # reshape performance metrics to a neat lil df
    mse_test = np.array(mse_test_li)
    mse_trai = np.array(mse_trai_li)
    df_test = pd.DataFrame({"mse" : mse_test})
    df_test['role'] = "test"
    df_trai = pd.DataFrame({"mse" : mse_trai})
    df_trai['role'] = "train"
    df_mse = pd.concat([df_test, df_trai], axis = 0)
    df_mse.shape

    # Save the model and all params 
    tstmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_name = tstmp + "_encoder_model_" + sess_info['model_gen'] + ".pth"
    torch.save(model_enc, os.path.join(path_trained_models, model_save_name))
    model_save_name = tstmp + "_decoder_model_" + sess_info['model_gen'] + ".pth"
    torch.save(model_dec, os.path.join(path_trained_models, model_save_name))
    di_sess = {'df_mse' : df_mse,'sess_info' : sess_info}
    sess_save_name = tstmp + "_session_info_" + sess_info['model_gen'] + ".pkl"
    with open(os.path.join(path_trained_models, sess_save_name), 'wb') as f:
        pickle.dump(di_sess, f)
            

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
    # organize as a dict
    out_di = {
        "feature_array" : feat,
        "image_file_name_array" : imfiles,
        "path_images" : path_images,
        "path_encoder" : path_enc,
        }
    return(out_di)


def wrap_to_dataset(di): 
    """
    Description:
    Arguments:
    """
    feat        = di['feature_array']
    imfiles     = di['image_file_name_array']
    path_images = di['path_images']
    path_enc    = di['path_encoder']
    # load metadata 
    path_xc_dir = os.path.dirname(path_images)
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
        'encoder_id' : path_enc,
        'meta': df_meta
        }
    with open(path_save_npz, 'wb') as handle:
        pickle.dump(dat_di, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # make zip of image dir 
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

    di = encoder_based_feature_extraction(path_images, path_models, time_stamp_model, devel = True)

    di['feature_array'].shape
    di['image_file_name_array'].shape
    di['path_images']
    di['path_encoder']

    wrap_to_dataset(di)


  

