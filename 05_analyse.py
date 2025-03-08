

#----------------------
#
#
#----------------------

import numpy as np
import pandas as pd
import plotly.express as px
import os 
# from sklearn.cluster import DBSCAN, AgglomerativeClustering
# from sklearn.preprocessing import StandardScaler
# from sklearn.manifold import Isomap
# import umap.umap_ as umap
from PIL import Image
# from sklearn.utils import shuffle

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


imgpath = "C:/xc_real_projects/xc_aec_project_sw_europe/downloaded_data_meta.pkl"

df_meta = pd.read_pickle(imgpath)  
df_meta.shape




path_clust = 'C:/xc_real_projects/clusters'
df_name = 'neigh_10_dims_16_eps_0.1400.pkl'
path_clu_df = os.path.join(path_clust, df_name)

df = pd.read_pickle(path_clu_df)  
df.shape




df_meta.columns


df_meta[['file', 'file-name']]

df_meta.head()


df.columns


df.head()






