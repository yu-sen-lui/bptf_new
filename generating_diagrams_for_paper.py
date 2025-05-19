# packages ========================================================================================
import numpy as np
import pandas as pd
import sparse
import os
import shutil
from tqdm import tqdm
import pickle
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

# folder management ===============================================================================
folder_path = "diagrams_for_paper"
if os.path.isdir(folder_path):
    print(f'{folder_path} exists')
else:
    os.mkdir(folder_path)
    print(f"{folder_path} created")
for entry in os.listdir(folder_path):
    path = os.path.join(folder_path, entry)
    if os.path.isfile(path) or os.path.islink(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)

# get tensor and namelists ========================================================================
assert os.path.isfile('sptensor.pkl') and os.path.isfile('name_lists.pkl'), 'Please run converting_to_sparse_tensors.py first'

with open('sptensor.pkl', 'rb') as f:
    data = pickle.load(f)
with open('name_lists.pkl', 'rb') as f:
    country_indices, action_indices, date_indices, database_indices = pickle.load(f)

# plot heatmap ====================================================================================
year = 2016
action = 3 # most common CAMEO code is 4, which is indexed as 3
database = database_indices[database_indices['database'] == 'GDELT'].iloc[0]['index']
number_of_countries = 20 # focus on top 100 actors

data_slice = data[:, :, action, year - 2000, database].todense()

ax = sns.heatmap(data_slice[:number_of_countries, :number_of_countries], linewidths=0, cmap='magma_r', xticklabels=False)
ax.set_yticks(np.arange(number_of_countries))
ax.set_yticklabels(country_indices['country'][:number_of_countries], fontsize=6, rotation=0)
ax.set_title(f"Top {number_of_countries} actors for {year} and CAMEO action {action_indices['action'].iloc[action]}")
plt.savefig(os.path.join(folder_path, 'heatmap.png'),
            bbox_inches='tight',
            dpi=600)

# overdispersion plot =============================================================================
