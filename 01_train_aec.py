#----------------------
# Author : Serge Zaugg
# Description : 
#----------------------

import plotly.express as px
import torch
import pandas as pd
import numpy as np
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
import os 
import datetime
import pickle 
from utils import SpectroImageDataset, make_data_augment_examples
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


path_untrained_models = "D:/xc_real_projects/untrained_models"

path_trained_models = "D:/xc_real_projects/trained_models"


sess_info = {
    'imgpath_train' : "D:/xc_real_projects/xc_ne_europe/images_24000sps_20250406_095331",
    'imgpath_test'  : "D:/xc_real_projects/xc_sw_europe/images_24000sps_20250406_092522",
    'batch_size_tr' : 8,
    'batch_size_te' : 32,
    'n_epochs' : 20,

    'hot_start' : False, 
    'model_tag' : 'gen_A',  

    # 'hot_start' : True, 
    # 'model_tag' : '20250511_094437',  

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
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=sess_info['batch_size_tr'],  shuffle=True, drop_last=True)

test_dataset  = SpectroImageDataset(sess_info['imgpath_test'], par = sess_info['data_generator'], augment_1 = False, denoise_1 = False, augment_2 = False, denoise_2 = True)
test_loader   = torch.utils.data.DataLoader(test_dataset, batch_size=sess_info['batch_size_te'],  shuffle=True, drop_last=True)



fig01 = make_data_augment_examples(pt_dataset = train_dataset, batch_size = 16)
fig01.show()

fig02 = make_data_augment_examples(pt_dataset = test_dataset, batch_size = 16)
fig02.show()

# get some info 
train_dataset.__len__()
test_dataset.__len__()
tra_itm = train_dataset.__getitem__(45)
tes_itm = test_dataset.__getitem__(45)
tra_itm[0].shape
tra_itm[1].shape
tes_itm[0].shape
tes_itm[1].shape




n_batches_tr = train_dataset.__len__() // sess_info['batch_size_tr']
n_batches_tr

n_batches_te = test_dataset.__len__() // sess_info['batch_size_te']
n_batches_te





# load  models 
if sess_info['hot_start'] == False:
    tstmp_0 = sess_info['model_tag']
    path_enc = [a for a in os.listdir(path_untrained_models) if tstmp_0 in a and 'cold_encoder' in a][0]
    path_dec = [a for a in os.listdir(path_untrained_models) if tstmp_0 in a and 'cold_decoder' in a][0]
    model_enc = torch.load(os.path.join(path_untrained_models, path_enc), weights_only = False)
    model_dec = torch.load(os.path.join(path_untrained_models, path_dec), weights_only = False)
elif sess_info['hot_start'] == True:
    tstmp_1 = sess_info['model_tag']
    path_enc = [a for a in os.listdir(path_trained_models) if tstmp_1 in a and 'encoder_model_' in a][0]
    path_dec = [a for a in os.listdir(path_trained_models) if tstmp_1 in a and 'decoder_model_' in a][0]
    model_enc = torch.load(os.path.join(path_trained_models, path_enc), weights_only = False)
    model_dec = torch.load(os.path.join(path_trained_models, path_dec), weights_only = False)
else:
    print("something is wrong with sess_info['hot_start']")





model_enc = model_enc.to(device)
summary(model_enc, (1, 128, 1152))

model_dec = model_dec.to(device)
summary(model_dec, (128, 1, 36))
summary(model_dec, (128, 1, 72))

# instantiate loss, optimizer
criterion = nn.MSELoss() #nn.BCELoss()
optimizer = optim.Adam(list(model_enc.parameters()) + list(model_dec.parameters()), lr=0.001)
# optimizer = optim.SGD(list(model_enc.parameters()) + list(model_dec.parameters()), lr=0.01, momentum=0.9)

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
    #----------------

    #----------------------------------
    # Testing the model at end of epoch 
    _ = model_enc.eval()
    _ = model_dec.eval()
    with torch.no_grad():
        test_perf_li = []
        for btchi, (da_te_1, da_te_2, fi) in enumerate(test_loader, 0):
            if btchi > 100: break
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

# fig_mse = px.line(
#     df_mse,
#     y = "mse",
#     color = "role"
#     )
# fig_mse.show()



# Save the model and all params 


tstmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

model_save_name = tstmp+"_encoder_model"+f"_epo_{epoch + 1}" +  ".pth"
torch.save(model_enc, os.path.join(path_trained_models, model_save_name))

model_save_name = tstmp+"_decoder_model"+f"_epo_{epoch + 1}" + ".pth"
torch.save(model_dec, os.path.join(path_trained_models, model_save_name))

di_sess = {'df_mse' : df_mse,'sess_info' : sess_info}
sess_save_name =   tstmp+"_session_info"+f"_epo_{epoch + 1}" + ".pkl"
with open(os.path.join(path_trained_models, sess_save_name), 'wb') as f:
    pickle.dump(di_sess, f)
        
# with open('saved_dictionary.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)


