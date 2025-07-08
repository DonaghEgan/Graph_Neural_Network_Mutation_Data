import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import process_data as prc
import download_study as ds

# msk_pan_2017
path, sources, urls = ds.download_study(name = 'msk_pan_2017')

# give path to process data 
data_dict = prc.read_files(path[0])

sample_embeddings = data_dict['sample_meta']['embeddings']
metadata = data_dict['sample_meta']['metadata']
sample_index = data_dict['sample_index']

# Apply UMAP to reduce dimensionality to 2D
reducer = umap.UMAP(n_components=2, init = 'random')
embeddings_2d = reducer.fit_transform(sample_embeddings)

#
df_meta = pd.DataFrame.from_dict(metadata, orient='index', columns=['TMB', 'ONCOTREE_CODE', 'CANCER_TYPE'])

# Create a DataFrame with embeddings and metadata
df_plot = pd.DataFrame(embeddings_2d, columns=['UMAP1', 'UMAP2'])
df_plot = df_plot.join(df_meta)

# Save the DataFrame to CSV for plotting in R
df_plot.to_csv('/home/degan/msk/data/umap_embedding_for_R.csv', index=False)