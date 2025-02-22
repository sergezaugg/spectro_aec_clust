#--------------------------------
# Assess trained models by direct comparison of a few reconstructed images
#--------------------------------



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




