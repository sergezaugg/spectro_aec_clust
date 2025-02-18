#----------------------
# Author : Serge Zaugg
# Description : 
#----------------------

import plotly.express as px
import torch
import plotly.express as px
import numpy as np
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
import os 
# from ptutils import SpectroImageDataset, Encoder, Decoder
from ptutils import SpectroImageDataset
from custom_models import Encoder, Decoder


torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

imgpath    = "C:/xc_real_projects/xc_aec_project/downloaded_data_img_24000sps_1ch"
model_path = "C:/xc_real_projects/models"


#----------------------
# define data loader 
train_dataset = SpectroImageDataset(imgpath)
train_dataset.__len__()
xx = train_dataset.__getitem__(45)
xx
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128,  shuffle=True)
for i, (data, fi) in enumerate(train_loader, 0):
    if i > 3:
        break
    print(data.shape)


#----------------------
# define models 

impsha = (128, 128)
latsha = 512
n_blck = 3

model_enc = Encoder(n_ch_in = 1, 
                    n_ch_latent=latsha, 
                    shape_input = impsha, 
                    n_conv_blocks = n_blck,
                    ch = [16, 32, 64, 256, 512],
                    po = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
                    ) 
model_enc = model_enc.to(device)
summary(model_enc, (1, 128, 128))

model_dec = Decoder(n_ch_out = 1, 
                    n_ch_latent=latsha, 
                    shape_output = impsha, 
                    n_conv_blocks = n_blck,
                    ch = [128, 64, 64, 32, 32],
                    po = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
                    )
model_dec = model_dec.to(device)
summary(model_dec, (latsha,))



# torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, 
#                  amsgrad=False, *, foreach=None, maximize=False, capturable=False, 
#                  differentiable=False, fused=None)


# instantiate loss, optimizer
criterion = nn.MSELoss() #nn.BCELoss()
optimizer = optim.Adam(list(model_enc.parameters()) + list(model_dec.parameters()), lr=0.001)
# optimizer = optim.Adam(list(model_enc.parameters()) + list(model_dec.parameters()), lr=0.0001)
# optimizer = optim.SGD(list(model_enc.parameters()) + list(model_dec.parameters()), lr=0.01, momentum=0.9)

_ = model_enc.train()
_ = model_dec.train()

n_epochs = 10

# initialize the best validation loss as infinity
best_val_loss = float("inf")
# start training by looping over the number of epochs
for epoch in range(n_epochs):
    print(f"Epoch: {epoch + 1}/{n_epochs}")
    # set the encoder and decoder models to training mode
   
    for btchi, (data, fi) in enumerate(train_loader, 0):
        # print(data.shape)
        data = data.to(device)
        # data.shape
        # reset the gradients 
        optimizer.zero_grad()
        # forward 
        encoded = model_enc(data).to(device)
        # encoded.shape
        decoded = model_dec(encoded).to(device)
        # decoded.shape
        # compute the reconstruction loss 
        loss = criterion(decoded, data)
        # compute the gradients
        loss.backward()
        # update the weights
        optimizer.step()
        # accumulate the loss 
    print('loss', np.round(loss.item(),5))
    print('data min max',    data.min().cpu().detach().numpy().round(4),     data.max().cpu().detach().numpy().round(4))
    print('decoded min max', decoded.min().cpu().detach().numpy().round(4),  decoded.max().cpu().detach().numpy().round(4)) 
    

# Save the model
torch.save(model_enc.state_dict(), os.path.join(model_path, "encoder_model_xx.pth") )
torch.save(model_enc.state_dict(), os.path.join(model_path, "encoder_model"+f"_epo_{epoch + 1}"+".pth") )







# check reconstruction with examples 
if False:

    data = data.to(device)
    encoded = model_enc(data).to(device)
    decoded = model_dec(encoded).to(device)
    data.shape

    # ii = 489 
    for ii in np.random.randint(data.shape[0], size = 15):
        img_orig = data[ii].cpu().numpy()
        img_orig.shape
        # img_orig = img_orig.transpose(1,2,0) # 3 ch
        img_orig = img_orig.squeeze() # 1 ch
        img_orig.min()
        img_orig.max()
        img_orig.dtype
        fig00 = px.imshow(img_orig, height = 500, title="original")
        fig00.show()

        img_reco = decoded[ii].cpu().detach().numpy()
        # img_reco = img_reco.transpose(1,2,0) # 3 ch
        img_reco = img_reco.squeeze()  # 1 ch
        img_reco.shape
        img_reco = 255*(img_reco - img_reco.min())/(img_reco.max())
        img_reco.min()
        img_reco.max()
        img_reco.dtype
        fig01 = px.imshow(img_reco, height = 500, title="reconstructed")
        fig01.show()




