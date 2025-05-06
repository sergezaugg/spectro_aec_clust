#----------------------
#
#
#----------------------

import numpy as np
import pandas as pd
import plotly.express as px
import os 
from sklearn.cluster import DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
import umap.umap_ as umap
from PIL import Image
from sklearn.utils import shuffle
import datetime


path_features = "C:/xc_real_projects/features"

imgpath = "C:/xc_real_projects/xc_streamlit_sw_eur/downloaded_data_img_24000sps_rect"

path_clust_images = "C:/xc_real_projects/clusters_01"

if not os.path.exists(path_clust_images):
    os.makedirs(path_clust_images)

path_save = os.path.join(path_features, 'features_medium20250319_210308.npz')


#-------------------------
# parame Reduce dimensionality
n_dims_red = 32 # 32 seems goooood 
n_neighbors = 10 # between 5-15 seems ok
# n_neighbors = 20 # between 5-15 seems ok

eps_search = np.arange(0.05, 0.5, 0.05) # defaults 32
# eps_search = np.arange(0.02, 0.5, 0.02) # 

dbscan_min_samples = 10
# dbscan_min_samples = 15
# dbscan_min_samples = 20

# centf01 = 4
# centf02 = 12

# # good 
# centf01 = 3
# centf02 = 13


#-------------------------
# (x) Load AEC features 
#-------------------------

# data = np.load(file = path_save)
# feat    = data['feat']
# imfiles = data['imfiles']
# feat.shape
# imfiles.shape

# # shuffle
# feat, imfiles = shuffle(feat, imfiles)
# feat.shape
# imfiles.shape


#-------------------------
# (x)
#-------------------------

# # atak only center time bins 
# feat.shape
# feat = feat[:, :,centf01:centf02] #  better 
# feat.shape

# # take a pooling function over time to get classic feature vector 
# # feature_mat = np.concatenate([feat.max(2),  feat.mean(2) , feat.std(2)], axis = 1) # default
# feature_mat = np.concatenate([feat.mean(2) , feat.std(2)], axis = 1) # also good 
# # feature_mat = feat.max(2)
# # feature_mat = feat.mean(2)
# feature_mat.shape

# # select valid features 
# feature_mat.std(0).shape
# sel_feats = feature_mat.std(0) != 0.000 # or feature_mat.std(0) != 0.0
# feature_mat_red = feature_mat[:,sel_feats]
# feature_mat_red.shape


# #-------------------------
# # (x) standardize 1
# #-------------------------

# scaler = StandardScaler()
# scaler.fit(feature_mat_red)
# feature_mat_scaled_1 = scaler.transform(feature_mat_red)
# # feature_mat_scaled_1.mean(0)
# # feature_mat_scaled_1.std(0)



# # got to 16 bit float to have small e npz files ()
# feature_mat_scaled_1 = feature_mat_scaled_1.astype('float16')


# # save smaller data just before clustering 

# tstmp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")



# path_save_mini = os.path.join(path_features, 'features_sr' + tstmp + '.npz')
# np.savez(file = path_save_mini, feat=feature_mat_scaled_1, imfiles=imfiles)


#-------------------------
# (x) Reduce dimensionality
#-------------------------
# 
if True:
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_dims_red, metric = 'euclidean')
    # reducer =    Isomap(n_neighbors=n_neighbors, n_components=n_dims_red, metric = "euclidean")
    reducer.fit(feature_mat_scaled_1[0:35000])
    X_trans = reducer.transform(feature_mat_scaled_1)
    X_trans.shape

if False:
    X_trans = feature_mat_scaled_1    
    X_trans.shape

#-------------------------
# (x) standardize 2
#-------------------------

scaler = StandardScaler()
scaler.fit(X_trans)
feature_mat_scaled_2 = scaler.transform(X_trans)
feature_mat_scaled_2.shape


# # save smaller data just before clustering 
# path_save_mini = os.path.join(path_features, 'features_full_reduced.npz')
# np.savez(file = path_save_mini, feat=feature_mat_scaled_2, imfiles=imfiles)


#-------------------------
# (x) Clustering
#-------------------------

for eps_i in eps_search:
    print("-----------")
    print(">> eps_i", eps_i)

    clu = DBSCAN(eps = eps_i, min_samples=dbscan_min_samples, metric = 'euclidean', n_jobs = 8) 

    # clu = OPTICS(min_samples = dbscan_min_samples, 
    #              max_eps = eps_i, 
    #              metric = 'euclidean', 
    #             #  cluster_method = 'dbscan', 
    #              cluster_method = 'xi', 
    #              n_jobs = 8, min_cluster_size = 30)
                                 

    cluster_ids = clu.fit_predict(feature_mat_scaled_2)
    pd.Series(cluster_ids).value_counts()[0:10]
 
    # select only large enough clusters 
    sel0 = pd.Series(cluster_ids).value_counts() > 10
    sel1 = pd.Series(cluster_ids).value_counts() < 500
    sel = np.logical_and(sel0, sel1)
    sel2 = pd.Series(cluster_ids).value_counts().loc[sel].index
    cluster_ids.shape

    try:
        cluster_ids_sel    = cluster_ids[pd.Series(cluster_ids).isin(sel2)]
        finle_name_arr_sel = imfiles[pd.Series(cluster_ids).isin(sel2)]
        cluster_ids_sel.shape
        finle_name_arr_sel.shape

        # prepare df 
        df = pd.DataFrame({
            'file_name' :finle_name_arr_sel,
            'cluster_id' :cluster_ids_sel,
            })
        df['newname'] = df['cluster_id'].astype(str).str.cat(others=df['file_name'], sep='_')
        df = df.sort_values(by = 'cluster_id')
        df.shape

        # MAKE NICE MOSAIK PLOT 
        imall = Image.new('L', (25000, 25000), '#55ff00')
        gap = 25
        current_id = df['cluster_id'].min()
        vert_counter = gap
        horiz_counter = gap
        for i,r in df.iterrows():
            im = Image.open(os.path.join(imgpath, r['file_name']))

            # #------------------------
            # # take only center part of image in time
            # w, h = im.size
            # # c1 = 64
            # # c2 = 64 + 128
            # c1 = 64 - 32 
            # c2 = 64 + 128 + 32
            # # im.crop((left, top, right, bottom))
            # im = im.crop((c1, 0, c2, h))
            # # im.size
            # #------------------------

            if r['cluster_id'] > current_id: 
                vert_counter = vert_counter + 128 + gap
                horiz_counter = 0 + gap
                current_id = r['cluster_id']
            imall.paste(im, ( horiz_counter, vert_counter))
            horiz_counter = horiz_counter + 128 + 128 + 64 + gap  # !!!!!!! added +128   
        # imall.show()


        str01 = 'neigh_' + "{:1.0f}".format(n_neighbors)  
        str02 = '_dims_' + "{:1.0f}".format(n_dims_red)  
        str03 = '_eps_' +   "{:5.4f}".format(eps_i) 
        str04 = '_min_samp_' +  "{:5.4f}".format(dbscan_min_samples)       
        # str05 = '_centf_' +  "{:5.4f}".format(centf01)  + "_{:5.4f}".format(centf02)  

        finam = str01 + str02 + str03 + str04 + str05

        imall.save(os.path.join(path_clust_images,  finam + ".png" ))
        df.to_pickle(path = os.path.join(path_clust_images,  finam + ".pkl" )  )

    except:
        print("haha")










