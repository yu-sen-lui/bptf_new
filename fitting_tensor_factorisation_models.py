# packages ========================================================================================
import sys
from IPython.display import display, Javascript

def restart_kernel():
    """Restart the Jupyter Notebook kernel to reflect changes in modules and packages."""
    display(Javascript("Jupyter.notebook.kernel.restart()"))
    print("Kernel is restarting...")

restart_kernel()

import bptf
from bptf import BPTF
import numpy as np
import pandas as pd
import sparse
import os
import shutil
from tqdm import tqdm
import pickle
import scipy.stats as st
import matplotlib.pyplot as plt
import torch
import tensorly
import cupy

import multiprocessing
from joblib import Parallel, delayed
from tqdm.contrib.concurrent import process_map

print(os.getcwd())

import gc
gc.collect()

# Get tensor ======================================================================================
assert os.path.exists('sptensor.pkl'), 'No such file.'
with open('sptensor.pkl', 'rb') as f:
    Y = pickle.load(f)

# Tensor factorisation ============================================================================
# decompose the tensor using Poisson CP decomposition
gc.collect()

include_mask = False
print(include_mask)

# # We need to mask the country-country diagonals as self actions are not included
# I_range = np.arange(Y.shape[0])
# A_range = np.arange(Y.shape[2])
# T_range = np.arange(Y.shape[3])
# D_range = np.arange(Y.shape[4])

# I, A, T, D = np.meshgrid(I_range, A_range, T_range, D_range, indexing='ij')
# I_flattened = np.ravel(I)
# A_flattened = np.ravel(A)
# T_flattened = np.ravel(T)
# D_flattened = np.ravel(D)

# mask_coords = np.vstack([I_flattened, I_flattened, A_flattened, T_flattened, D_flattened])
# mask = sparse.COO(coords=mask_coords, data=np.ones(shape = mask_coords.shape[1]), shape=Y.shape) if include_mask else None
# print(include_mask)

# need to enforce self-country actions = 0
mask = Y.coords[0] != Y.coords[1]
new_coords = Y.coords[:, mask].copy()
new_data = Y.data[mask].copy()
Y = sparse.COO(new_coords, new_data, shape=Y.shape)

# need to mask unobserved dates for ICEWS and TERRIER
# ICEWS is missing data from 2024 (1 year)
# TERRIER is missing data from 2019 onwards (5 years)
#   database  index
# 0    ICEWS      0
# 1    GDELT      1
# 2  TERRIER      2
# ICEWS (D = )
I_range = np.arange(Y.shape[0])
A_range = np.arange(Y.shape[1])
T_range = np.arange(Y.shape[2])
D_range = np.arange(Y.shape[3])

ICEWS_masked_months = T_range[-12]
I, J, A, T, D = np.meshgrid()

n_components = 100
# fitting without mask, just doing an inner join of all 3 datasets
Y_2000_2018 = Y[:, :, :, :(12*(2019-2000)), :]
bptf_5mode = BPTF(data_shape=Y_2000_2018.shape, n_components=n_components)
bptf_5mode.fit(Y_2000_2018, max_iter = 50, mask=None,verbose=True)
print('Fitting with BPTF')
with open('bptf_5mode_50iter_2000_2018.pkl', 'wb') as f:
    pickle.dump(bptf_5mode, f)

# check the shapes
for j in range(len(Y.shape)):
    assert bptf_5mode.G_DK_M[j].shape == (Y.shape[j], n_components)

del bptf_5mode
gc.collect()

# fitting with mask

# Tensor factorisation with deterministic method ==================================================
# if os.path.exists('nntf_parafac_5mode.pkl'):
#     with open('nntf_parafac_5mode.pkl', 'rb') as f:
#         tensor_mu = pickle.load(f)
#     print('File exists')
# else:
#     print('Fitting with deterministic algorithm')
#     cp_init = tensorly.cp_tensor.CPTensor(
#         tensorly.decomposition._cp.initialize_cp(
#             Y, non_negative = True, init = 'random', rank = n_components
#             )
#         )
#     tensor_mu, _ = tensorly.decomposition.non_negative_parafac(
#         Y, rank=n_components, init=cp_init, return_errors=True
#         )
    
#     with open('nntf_parafac_5mode.pkl', 'wb') as f:
#         pickle.dump(tensor_mu, f)