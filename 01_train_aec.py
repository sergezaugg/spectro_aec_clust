#----------------------
# Author : Serge Zaugg
# Description : 
#----------------------

import plotly.express as px
import torch
import numpy as np
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
import os 
import datetime
from custom_models import SpectroImageDataset
from custom_models import EncoderSimple, DecoderTransp, DecoderUpsample
# import pickle
from plotly.subplots import make_subplots


torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

imgpath_train = "C:/xc_real_projects/xc_aec_project_n_europe/downloaded_data_img_24000sps"

model_path = "C:/xc_real_projects/models"

batch_size = 64

n_epochs = 3

#----------------------
# define data loader 
train_dataset = SpectroImageDataset(imgpath_train)
train_dataset.__len__()
xx = train_dataset.__getitem__(45)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,  shuffle=True, drop_last=True)
for i, (da_orig, data_augm, fi) in enumerate(train_loader, 0):
    if i > 3:
        break
    print(data_augm.shape)
    print(da_orig.shape)


n_batches = train_dataset.__len__() // batch_size
n_batches



#----------------------
# define models 


model_enc = EncoderSimple()
model_enc = model_enc.to(device)
summary(model_enc, (1, 128, 128))


# model_dec = DecoderTransp()
model_dec = DecoderUpsample()

model_dec = model_dec.to(device)
summary(model_dec, (512, 1, 8))

   



_ = model_enc.train()
_ = model_dec.train()




# --------------------------------
# torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, 
#                  amsgrad=False, *, foreach=None, maximize=False, capturable=False, differentiable=False, fused=None)

# instantiate loss, optimizer
criterion = nn.MSELoss() #nn.BCELoss()
optimizer = optim.Adam(list(model_enc.parameters()) + list(model_dec.parameters()), lr=0.001)
# optimizer = optim.SGD(list(model_enc.parameters()) + list(model_dec.parameters()), lr=0.01, momentum=0.9)

for epoch in range(n_epochs):
    print(f"Epoch: {epoch + 1}/{n_epochs}")
    # set the encoder and decoder models to training mode
    loss_tra =[]
    for btchi, (da_orig, data_augm, fi) in enumerate(train_loader, 0):
        # if btchi > 5000:
        #     break
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
        # compute the gradients
        loss.backward()
        # update the weights
        optimizer.step()

        if btchi % 10 == 0:
            print('loss', np.round(loss.item(),5), "   --- status: "  + str(btchi) + " out of " + str(n_batches) + " batches")
            print(decoded.cpu().detach().numpy().min().round(3) , decoded.cpu().detach().numpy().max().round(3) )
            print("-")

  
    
# Save the model
if True:
    tstmp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")

    model_save_name = "encoder_model"+tstmp+f"_epo_{epoch + 1}" +  ".pth"
    torch.save(model_enc.state_dict(), os.path.join(model_path, model_save_name))

    model_save_name = "decoder_model"+tstmp+f"_epo_{epoch + 1}" + ".pth"
    torch.save(model_dec.state_dict(), os.path.join(model_path, model_save_name))

    # param_save_name = "params_model"+tstmp+f"_epo_{epoch + 1}" + ".json"
    # with open(os.path.join(model_path, param_save_name), 'wb') as fp:
    #     pickle.dump(par, fp)





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
        _ = fig.show()





