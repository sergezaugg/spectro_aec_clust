#--------------------------------
# Author : Serge Zaugg
# Description : small helper functions
# Description : bigger ML processes are wrapped into functions here
#--------------------------------

import os 
import pickle
import numpy as np
import pandas as pd
import shutil
import datetime
from PIL import Image
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms.v2 as transforms
import torch.optim as optim
import json
from torchsummary import summary

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SpectroImageDataset(Dataset):

    def __init__(self, imgpath, par = None, augment_1=False, augment_2=False, denoise_1=False, denoise_2=False):
        self.all_img_files = [a for a in os.listdir(imgpath) if '.png' in a]
        self.imgpath = imgpath
        self.par = par
        self.augment_1 = augment_1
        self.augment_2 = augment_2
        self.denoise_1 = denoise_1
        self.denoise_2 = denoise_2

        if self.augment_1 or self.augment_2 or self.denoise_1 or self.denoise_2:
            self.dataaugm = transforms.Compose([
                transforms.RandomAffine(translate=(self.par['da']['trans_prop'], 0.0), degrees=(-self.par['da']['rot_deg'], self.par['da']['rot_deg'])),
                transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianNoise(mean = 0.0, sigma = self.par['da']['gnoisesigm'], clip=True),]), p=self.par['da']['gnoiseprob']),
                transforms.ColorJitter(brightness = self.par['da']['brightness'] , contrast = self.par['da']['contrast']),
                ])
 
    def __getitem__(self, index):     
        img = Image.open( os.path.join(self.imgpath,  self.all_img_files[index] ))
        # load pimage and set range to [0.0, 1.0]
        x_1 = pil_to_tensor(img).to(torch.float32) / 255.0
        x_2 = pil_to_tensor(img).to(torch.float32) / 255.0
        # simple de-noising with threshold
        if self.denoise_1: 
            x_1[x_1 < self.par['den']['thld'] ] = 0.0
        if self.denoise_2: 
            x_2[x_2 < self.par['den']['thld'] ] = 0.0    
        # data augmentation 
        if self.augment_1: 
            x_1 = self.dataaugm(x_1)  
        if self.augment_2:
            x_2 = self.dataaugm(x_2) 
        # prepare meta-data 
        y = self.all_img_files[index]

        return (x_1, x_2, y)
    
    def __len__(self):
        return (len(self.all_img_files))



class AutoencoderTrain:
  
    def __init__(self, sess_json):
        """
        sess_json : name of one of the session configuration json files that are stored in ./session_params
        """
        with open(os.path.join('./session_params', sess_json )) as f:
            sess_info = json.load(f)
        self.sess_info = sess_info    
        self.train_dataset = SpectroImageDataset(self.sess_info['imgpath_train'], par = self.sess_info['data_generator'], augment_1 = True, denoise_1 = False, augment_2 = False, denoise_2 = True)
        self.test_dataset  = SpectroImageDataset(self.sess_info['imgpath_test'],  par = self.sess_info['data_generator'], augment_1 = False, denoise_1 = False, augment_2 = False, denoise_2 = True)
  

        if sess_info['hot_start'] == False:
            tstmp_0 = sess_info['model_tag']
            path_enc = [a for a in os.listdir(sess_info['path_untrained_models']) if tstmp_0 in a and 'cold_encoder' in a][0]
            path_dec = [a for a in os.listdir(sess_info['path_untrained_models']) if tstmp_0 in a and 'cold_decoder' in a][0]
            self.model_enc = torch.load(os.path.join(sess_info['path_untrained_models'], path_enc), weights_only = False)
            self.model_dec = torch.load(os.path.join(sess_info['path_untrained_models'], path_dec), weights_only = False)
            sess_info['model_gen'] = sess_info['model_tag']
        elif sess_info['hot_start'] == True:
            tstmp_1 = sess_info['model_tag']
            path_enc = [a for a in os.listdir(sess_info['path_trained_models']) if tstmp_1 in a and 'encoder_model' in a][0]
            path_dec = [a for a in os.listdir(sess_info['path_trained_models']) if tstmp_1 in a and 'decoder_model' in a][0]
            self.model_enc = torch.load(os.path.join(sess_info['path_trained_models'], path_enc), weights_only = False)
            self.model_dec = torch.load(os.path.join(sess_info['path_trained_models'], path_dec), weights_only = False) 
            # load info from previous training session 
            path_sess = [a for a in os.listdir(sess_info['path_trained_models']) if tstmp_1 in a and '_session_info' in a][0]
            with open(os.path.join(sess_info['path_trained_models'], path_sess), 'rb') as f:
                di_origin_sess = pickle.load(f)
            # load model generation 
            sess_info['model_gen'] = di_origin_sess['sess_info']['model_gen']
        else:
            print("something is wrong with sess_info['hot_start']")
        # return(model_enc, model_dec)


    def make_data_augment_examples(self, batch_size = 12):
        """
        # assess a realization of data augmentation 
        pt_dataset : an instance of torch.utils.data.Dataset
        """
        pt_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size,  shuffle=False, drop_last=True)
        # take only first batch 
        for i, (da_1, da_2, fi) in enumerate(pt_loader, 0):
            if i > 0: break
        fig = make_subplots(rows=batch_size, cols=2)
        for ii in range(batch_size): 
            # print(ii)
            img_1 = da_1[ii].cpu().detach().numpy()
            # img_1 = img_1.squeeze() # 1 ch
            img_1 = np.moveaxis(img_1, 0, 2) # 3 ch
            img_1 = 255*img_1 
            img_2 = da_2[ii].cpu().detach().numpy()
            # img_2 = img_2.squeeze()  # 1 ch
            img_2 = np.moveaxis(img_2, 0, 2) # 3 ch
            img_2 = 255*img_2 
            fig.add_trace(px.imshow(img_1).data[0], row=ii+1, col=1)
            fig.add_trace(px.imshow(img_2).data[0], row=ii+1, col=2)
        fig.update_layout(autosize=True,height=400*batch_size, width = 2000)
        return(fig)
    

    def train_autoencoder(self, devel = False):

        # train 
        train_loader  = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.sess_info['batch_size_tr'],  shuffle=True, drop_last=True)
        test_loader   = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.sess_info['batch_size_te'],  shuffle=True, drop_last=True)

        # instantiate loss and optimizer
        criterion = nn.MSELoss() #nn.BCELoss()
        optimizer = optim.Adam(list(self.model_enc.parameters()) + list(self.model_dec.parameters()), lr=0.001)
        # optimizer = optim.SGD(list(self.model_enc.parameters()) + list(self.model_dec.parameters()), lr=0.01, momentum=0.9)

        n_batches_tr = self.train_dataset.__len__() // self.sess_info['batch_size_tr']
        n_batches_te = self.test_dataset.__len__() // self.sess_info['batch_size_te']

        mse_test_li = []
        mse_trai_li = []
        for epoch in range(self.sess_info['n_epochs']):
            print(f"Epoch: {epoch + 1}/{self.sess_info['n_epochs']}")
            #----------------
            # Train the model 
            _ = self.model_enc.train()
            _ = self.model_dec.train()
            trai_perf_li = []
            for batch_tr, (da_tr_1, da_tr_2, fi) in enumerate(train_loader, 0):
                if devel and batch_tr > 1:
                    break
                da_tr_1 = da_tr_1.to(device)
                da_tr_2 = da_tr_2.to(device)
                # reset the gradients 
                optimizer.zero_grad()
                # forward 
                encoded = self.model_enc(da_tr_1)
                # encoded.shape
                decoded = self.model_dec(encoded)
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
            _ = self.model_enc.eval()
            _ = self.model_dec.eval()
            with torch.no_grad():
                test_perf_li = []
                for btchi, (da_te_1, da_te_2, fi) in enumerate(test_loader, 0):
                    if btchi > 10: break # 100
                    da_te_1 = da_te_1.to(device)
                    da_te_2 = da_te_2.to(device)
                    # forward 
                    encoded = self.model_enc(da_te_1)#.to(device)
                    # encoded.shape
                    decoded = self.model_dec(encoded)#.to(device)
                    # compute the reconstruction loss 
                    loss_test = criterion(decoded, da_te_2)
                    test_perf_li.append(loss_test.cpu().detach().numpy().item())
                    # feedback every 10th batch
                    if btchi % 10 == 0:
                        print('TEST loss', np.round(loss_test.item(),5), " --- "  + str(btchi) + " out of " + str(n_batches_te) + " batches")
                mse_test_li.append(np.array(test_perf_li).mean())
            
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
        model_save_name = tstmp + "_encoder_model_" + self.sess_info['model_gen'] + ".pth"
        torch.save(self.model_enc, os.path.join(self.sess_info['path_trained_models'], model_save_name))
        model_save_name = tstmp + "_decoder_model_" + self.sess_info['model_gen'] + ".pth"
        torch.save(self.model_dec, os.path.join(self.sess_info['path_trained_models'], model_save_name))
        di_sess = {'df_mse' : df_mse,'sess_info' : self.sess_info}
        sess_save_name = tstmp + "_session_info_" + self.sess_info['model_gen'] + ".pkl"
        with open(os.path.join(self.sess_info['path_trained_models'], sess_save_name), 'wb') as f:
            pickle.dump(di_sess, f)

        # save model for external projects    
        model_save_name = tstmp + "_encoder_script_" + self.sess_info['model_gen'] + ".pth"
        model_enc_scripted = torch.jit.script(self.model_enc) # Export to TorchScript
        model_enc_scripted.save(os.path.join(self.sess_info['path_trained_models'], model_save_name))   








def dim_reduce(X, n_neigh, n_dims_red):
    """
    Conveniant wrapper around UMAP dim reduction with pre and post scaling
    """
    scaler = StandardScaler()
    reducer = umap.UMAP(
        n_neighbors = n_neigh, 
        n_components = n_dims_red, 
        metric = 'euclidean',
        n_jobs = -1
        )
    X_scaled = scaler.fit_transform(X)
    X_trans = reducer.fit_transform(X_scaled)
    X_out = scaler.fit_transform(X_trans)
    return(X_out)






class AutoencoderExtract:
  
    def __init__(self, path_images, path_models, time_stamp_model):  
        """
        Arguments :
            path_images : 
            path_models : 
            time_stamp_model :
        """    
        self.path_images = path_images
        self.path_models = path_models
        self.time_stamp_model = time_stamp_model

    def evaluate_reconstruction_on_examples(self, n_images = 16):
        """
        Assess trained models by direct comparison of a few reconstructed images
        """
        # ---------------------
        # (1) load a few images 
        test_dataset = SpectroImageDataset(self.path_images, augment_1 = False, denoise_1 = False, augment_2 = False, denoise_2 = False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = n_images, shuffle = True)
        for i_test, (data_1, data_2 , _ ) in enumerate(test_loader, 0):
            if i_test > 0: break
            print(data_1.shape)
            print(data_2.shape)
        # ---------------------
        # (2) load models  
        path_enc = [a for a in os.listdir(self.path_models) if self.time_stamp_model in a and 'encoder_model' in a][0]
        path_dec = [a for a in os.listdir(self.path_models) if self.time_stamp_model in a and 'decoder_model' in a][0]
        # load trained AEC
        model_enc = torch.load( os.path.join(self.path_models, path_enc), weights_only = False)
        model_dec = torch.load( os.path.join(self.path_models, path_dec), weights_only = False)
        model_enc = model_enc.to(device)
        model_dec = model_dec.to(device)
        _ = model_enc.eval()
        _ = model_dec.eval()
        # ---------------------
        # (3) predict 
        data = data_1.to(device)
        encoded = model_enc(data).to(device)
        decoded = model_dec(encoded).to(device)
        # ---------------------
        # plot 
        fig = make_subplots(rows=n_images, cols=2,)
        for ii in range(n_images) : 
            img_orig = data_2[ii].cpu().numpy()
            # img_orig = img_orig.squeeze() # 1 ch
            img_orig = np.moveaxis(img_orig, 0, 2) # 3 ch
            img_orig = 255.0*img_orig  
            img_reco = decoded[ii].cpu().detach().numpy()
            # img_reco = img_reco.squeeze()  # 1 ch
            img_reco = np.moveaxis(img_reco, 0, 2) # 3 ch
            img_reco = 255.0*img_reco   
            _ = fig.add_trace(px.imshow(img_orig).data[0], row=ii+1, col=1)
            _ = fig.add_trace(px.imshow(img_reco).data[0], row=ii+1, col=2)
        _ = fig.update_layout(autosize=True,height=400*n_images, width = 800)
        _ = fig.update_layout(title="Model ID: " + self.time_stamp_model)
        return(fig)

    def encoder_based_feature_extraction(self, batch_size = 128, shuffle = True, devel = False):
        """
        Description: Applies a trained encoder to images in a dir and extracts the latent representation as a 2D feature array
        Arguments:
        """
        # get the file corresponding to the time stamp
        path_enc = [a for a in os.listdir(self.path_models) if self.time_stamp_model in a and 'encoder_model' in a][0]
        # load trained AEC
        model_enc = torch.load(os.path.join(self.path_models, path_enc), weights_only = False)
        model_enc = model_enc.to(device)
        _ = model_enc.eval()
        # prepare dataloader
        test_dataset = SpectroImageDataset(self.path_images, augment_1 = False, denoise_1 = False, augment_2 = False, denoise_2 = False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = shuffle)
        # extract features
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
            if devel and i > 2:
                break
        # transform lists to array 
        feat = np.concatenate(feat_li)
        feat = feat.squeeze()
        imfiles = np.concatenate(imfiles)
        # save as npz
        tag = '_'.join(os.path.basename(path_enc).split('_')[0:2])     
        out_name = os.path.join(os.path.dirname(self.path_images), 'full_features_' + 'saec_' + tag + '.npz')
        np.savez(file = out_name, X = feat, N = imfiles)

    def time_pool_and_dim_reduce(self, n_neigh = 10):
        """
        """
        npzfile_full_path = os.path.join(os.path.dirname(self.path_images), 'full_features_' + 'saec_' + self.time_stamp_model + '.npz')
        file_name_in = os.path.basename(npzfile_full_path)        
        # load full features 
        npzfile = np.load(npzfile_full_path)
        X = npzfile['X']
        N = npzfile['N']
        # combine information over time
        # cutting time edges (currently hard coded to 20% on each side)
        ecut = np.ceil(0.10 * X.shape[2]).astype(int)
        X = X[:, :, ecut:(-1*ecut)] 
        print('Feature dim After cutting time edges:', X.shape)
        # full average pool over time 
        X_mea = X.mean(axis=2)
        X_std = X.std(axis=2)
        X_mea.shape
        X_std.shape
        X = np.concatenate([X_mea, X_std], axis = 1)
        print('Feature dim After average/std pool along time:', X.shape)
        # X.shape
        # N.shape
        # make 2d feats needed for plot 
        X_2D = dim_reduce(X, n_neigh, 2)
        for n_dims_red in [2,4,8,16, 32]:
            X_red = dim_reduce(X, n_neigh, n_dims_red)
            print(X.shape, X_red.shape, X_2D.shape, N.shape)
            # save as npz
            tag_dim_red = "dimred_" + str(n_dims_red) + "_neigh_" + str(n_neigh) + "_"
            file_name_out = tag_dim_red + '_'.join(file_name_in.split('_')[2:5])
            out_name = os.path.join(os.path.dirname(npzfile_full_path), file_name_out)
            np.savez(file = out_name, X_red = X_red, X_2D = X_2D, N = N)

            





# devel 
if __name__ == "__main__":
    print(22)


  




