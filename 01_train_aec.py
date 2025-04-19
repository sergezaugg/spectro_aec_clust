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
from utils import SpectroImageDataset
from custom_models_2 import EncoderAvgpool, DecoderTranspNew
from plotly.subplots import make_subplots


torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

imgpath_train = "D:/xc_real_projects/xc_ne_europe/images_24000sps_20250406_095331"

imgpath_test  = "D:/xc_real_projects/xc_sw_europe/images_24000sps_20250406_092522"

model_path = "D:/xc_real_projects/models"

batch_size = 8

n_epochs = 20

#----------------------
# define data loader 
train_dataset = SpectroImageDataset(imgpath_train, edge_attenuation = False, do_augment = True)
train_dataset.__len__()
xx = train_dataset.__getitem__(45)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,  shuffle=True, drop_last=True)
for i, (da_orig, data_augm, fi) in enumerate(train_loader, 0):
    if i > 2:
        break
    print(data_augm.shape)
    print(da_orig.shape)

n_batches = train_dataset.__len__() // batch_size
n_batches


test_dataset = SpectroImageDataset(imgpath_test, edge_attenuation = False, do_augment = False)
test_dataset.__len__()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,  shuffle=True, drop_last=True)

#----------------------
# define untrained models 
model_enc = EncoderAvgpool()
model_dec = DecoderTranspNew()

#----------------------
# load pre-trained  models 
if True: 
    tstmp_old = '20250418_223302'
    epotag = '_epo_14'
    path_enc = 'encoder_model_' + tstmp_old + epotag + '.pth'
    path_dec = 'decoder_model_' + tstmp_old + epotag + '.pth'
    model_enc.load_state_dict(torch.load(os.path.join(model_path, path_enc), weights_only=True))
    model_dec.load_state_dict(torch.load(os.path.join(model_path, path_dec), weights_only=True))

# encoder_model_20250418_223302_epo_14.pth
# encoder_model_20250418_223302_epo_14.pth


model_enc = model_enc.to(device)
# summary(model_enc, (1, 128, 128))
summary(model_enc, (1, 128, 1152))

model_dec = model_dec.to(device)
# summary(model_dec, (512, 1, 8))
# summary(model_dec, (512, 1, 72))
summary(model_dec, (128, 1, 36))

# instantiate loss, optimizer
criterion = nn.MSELoss() #nn.BCELoss()
optimizer = optim.Adam(list(model_enc.parameters()) + list(model_dec.parameters()), lr=0.001)
# optimizer = optim.SGD(list(model_enc.parameters()) + list(model_dec.parameters()), lr=0.001, momentum=0.9)

mse_test_li = []
mse_trai_li = []
for epoch in range(n_epochs):
    print(f"Epoch: {epoch + 1}/{n_epochs}")
    # set the encoder and decoder models to training mode
    loss_tra =[]
    _ = model_enc.train()
    _ = model_dec.train()
    trai_perf_li = []
    for btchi, (da_orig, data_augm, fi) in enumerate(train_loader, 0):
        if btchi > 700:
            break
        # print(btchi)
        data_augm = data_augm.to(device)
        da_orig = da_orig.to(device)
        # reset the gradients 
        optimizer.zero_grad()
        # forward 
        encoded = model_enc(data_augm)#.to(device)
        # encoded.shape
        decoded = model_dec(encoded)#.to(device)
        # compute the reconstruction loss 
        loss = criterion(decoded, da_orig)
        trai_perf_li.append(loss.cpu().detach().numpy().item())
        # compute the gradients
        loss.backward()
        # update the weights
        optimizer.step()

        if btchi % 10 == 0:
            print('loss', np.round(loss.item(),5), "   --- status: "  + str(btchi) + " out of " + str(n_batches) + " batches")
            print(decoded.cpu().detach().numpy().min().round(3) , decoded.cpu().detach().numpy().max().round(3) )
            print("-")
    mse_trai_li.append(np.array(trai_perf_li).mean())        


    # Testing the model
    _ = model_enc.eval()
    _ = model_dec.eval()
    with torch.no_grad():
        test_perf_li = []
        for btchi, (da_orig, data_augm, fi) in enumerate(test_loader, 0):
            if btchi > 20:
                break
             # print(btchi)
            data_augm = data_augm.to(device)
            da_orig = da_orig.to(device)
            # forward 
            encoded = model_enc(data_augm)#.to(device)
            # encoded.shape
            decoded = model_dec(encoded)#.to(device)
            # compute the reconstruction loss 
            loss_test = criterion(decoded, da_orig)
            test_perf_li.append(loss_test.cpu().detach().numpy().item())
            if btchi % 10 == 0:
                print('TEST loss', np.round(loss_test.item(),5), "   --- status: "  + str(btchi) )
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


fig_mse = px.line(
    df_mse,
    y = "mse",
    color = "role"
    )
fig_mse.show()






# Save the model
if True:
    tstmp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")

    model_save_name = "encoder_model"+tstmp+f"_epo_{epoch + 1}" +  ".pth"
    torch.save(model_enc.state_dict(), os.path.join(model_path, model_save_name))

    model_save_name = "decoder_model"+tstmp+f"_epo_{epoch + 1}" + ".pth"
    torch.save(model_dec.state_dict(), os.path.join(model_path, model_save_name))

    mse_save_name = "df_mse"+tstmp+f"_epo_{epoch + 1}" + ".pkl"
    df_mse.to_pickle(path=os.path.join(model_path, mse_save_name))


# check reconstruction with examples 
if True:

    _ = model_enc.eval()
    _ = model_dec.eval()

    data = da_orig.to(device)
    encoded = model_enc(data).to(device)
    decoded = model_dec(encoded).to(device)

    # ii = 489 
    for ii in np.random.randint(data.shape[0], size = 15):
        img_orig = data[ii].cpu().numpy()
        img_orig = img_orig.squeeze() # 1 ch
        img_orig = 255*(img_orig - img_orig.min())/(img_orig.max())
        # fig00 = px.imshow(img_orig, height = 500, title="original")
        img_reco = decoded[ii].cpu().detach().numpy()
        img_reco = img_reco.squeeze()  # 1 ch
        img_reco = 255*(img_reco - img_reco.min())/(img_reco.max())
        # fig01 = px.imshow(img_reco, height = 500, title="reconstructed")
        fig = make_subplots(rows=1, cols=2)
        _ = fig.add_trace(px.imshow(img_orig).data[0], row=1, col=1)
        _ = fig.add_trace(px.imshow(img_reco).data[0], row=1, col=2)
        _ = fig.update_layout(autosize=True,height=550, width = 1000)
        fig.show()





