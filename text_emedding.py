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
reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(sample_embeddings)

#
df_meta = pd.DataFrame.from_dict(metadata, orient='index', columns=['TMB', 'ONCOTREE_CODE', 'CANCER_TYPE'])

# Create a DataFrame with embeddings and metadata
df_plot = pd.DataFrame(embeddings_2d, columns=['UMAP1', 'UMAP2'])
df_plot = df_plot.join(df_meta)

fig = plt.figure(figsize=(6, 4)) # Get the figure object
ax = fig.add_subplot(111) # Get the axes object

sns.scatterplot(
    data=df_plot,
    x='UMAP1',
    y='UMAP2',
    hue='CANCER_TYPE',
    palette='tab10',
    s=20,
    alpha=0.7,
    ax=ax # Pass the axes object to seaborn
)

ax.set_title('UMAP of Sample Embeddings', fontsize=8)
ax.legend(
    loc='upper left',
    borderaxespad=0,
    fontsize='small',
    title='Cancer Type',
    title_fontsize='small',
    ncol=3
)

plt.savefig('/home/degan/msk/figures/umap_sample_embedding.png', bbox_inches = 'tight', dpi=300) # No bbox_inches='tight' if using this method primarily
plt.show()