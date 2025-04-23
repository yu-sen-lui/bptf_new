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

def check_mask(tensor):
    if not isinstance(tensor, np.ndarray):
        tensor = tensor.todense()
    assert ((tensor == 0) | (tensor == 1)).all()

# Get tensor ======================================================================================
assert os.path.exists('sptensor.pkl'), 'No such file.'
with open('sptensor.pkl', 'rb') as f:
    Y = pickle.load(f)

# Tensor factorisation ============================================================================
# decompose the tensor using Poisson CP decomposition
gc.collect()

include_mask = True
if include_mask:
    print('Including mask')
else:
    print('Not including mask')

# need to enforce self-country actions = 0
# mask = Y.coords[0] != Y.coords[1]
# new_coords = Y.coords[:, mask].copy()
# new_data = Y.data[mask].copy()
# Y = sparse.COO(new_coords, new_data, shape=Y.shape)

n_components = 50
max_iter = 500
# fitting an inner join of all 3 datasets
Y_2000_2018 = Y[:, :, :, :(12*(2019-2000)), :]

# we need a mask for the aprils of GDELT here as well
if include_mask:
    # For GDELT, we have an issue with GDELT1
    # GDELT1 spans up to 2014, and April of each year has an abnormally large count, about 5 times the other months
    I_range = np.arange(Y_2000_2018.shape[0])
    A_range = np.arange(Y_2000_2018.shape[2])
    T_range = np.arange(Y_2000_2018.shape[3])
    D_range = np.arange(Y_2000_2018.shape[4])

    select_aprils = list(range(3, 12*(2015-2000), 12))
    GDELT_masked_months = np.array(select_aprils)
    coordinates = np.meshgrid(I_range, I_range, A_range, GDELT_masked_months, np.ones(1))
    flattened_indices = [np.ravel(coords) for coords in coordinates]
    flattened_indices = np.vstack(flattened_indices)
    flattened_indices = flattened_indices.astype(np.int64)
    GDELT_mask = sparse.COO(coords=flattened_indices, data=np.ones(flattened_indices.shape[1]), shape=Y_2000_2018.shape)
    GDELT_mask = GDELT_mask.todense()
    GDELT_mask[np.eye(GDELT_mask.shape[0]).astype(bool)] = 1
    GDELT_mask = sparse.COO(GDELT_mask.astype(np.int64))
    # GDELT_mask = (1 - GDELT_mask.todense()).astype(np.int64)
    # GDELT_mask = sparse.COO(GDELT_mask.copy())
else:
    GDELT_mask = None

bptf_5mode = BPTF(data_shape=Y_2000_2018.shape, n_components=n_components)
filepath = f'bptf_5mode_{max_iter}_{n_components}_components_iter_2000_2018.pkl'
if not os.path.exists(filepath):
    print('Fitting with BPTF')
    bptf_5mode.fit(Y_2000_2018, max_iter = max_iter, mask=GDELT_mask, verbose=True, missing_val=1)
    with open(filepath, 'wb') as f:
        pickle.dump(bptf_5mode, f)
else:
    print(f'{filepath} exists')
    with open(filepath, 'rb') as f:
        bptf_5mode = pickle.load(f)

# check the shapes
for j in range(len(Y_2000_2018.shape)):
    assert bptf_5mode.G_DK_M[j].shape == (Y_2000_2018.shape[j], n_components)

del bptf_5mode
gc.collect()

# fitting with mask
if include_mask:
    # need to mask unobserved dates for ICEWS and TERRIER
    # ICEWS is missing data from 2024 (1 year)
    # TERRIER is missing data from 2019 onwards (5 years)
    #   database  index
    # 0    ICEWS      0
    # 1    GDELT      1
    # 2  TERRIER      2
    # ICEWS 
    I_range = np.arange(Y.shape[0])
    A_range = np.arange(Y.shape[2])
    T_range = np.arange(Y.shape[3])
    D_range = np.arange(Y.shape[4])

    ICEWS_masked_months = np.arange(Y.shape[3])[-12:]
    coordinates = np.meshgrid(I_range, I_range, A_range, ICEWS_masked_months, np.zeros(1))
    flattened_indices = [np.ravel(coords) for coords in coordinates]
    flattened_indices = np.vstack(flattened_indices)
    flattened_indices = flattened_indices.astype(np.int64)
    ICEWS_mask = sparse.COO(coords=flattened_indices, data=np.ones(flattened_indices.shape[1]), shape=Y.shape)

    # TERRIER
    TERRIER_masked_months = np.arange(Y.shape[3])[-5*12:]
    coordinates = np.meshgrid(I_range, I_range, A_range, TERRIER_masked_months, np.ones(1)*2)
    flattened_indices = [np.ravel(coords) for coords in coordinates]
    flattened_indices = np.vstack(flattened_indices)
    flattened_indices = flattened_indices.astype(np.int64)
    TERRIER_mask = sparse.COO(coords=flattened_indices, data=np.ones(flattened_indices.shape[1]), shape=Y.shape)

    # For GDELT, we have an issue with GDELT1
    # GDELT1 spans up to 2014, and April of each year has an abnormally large count, about 5 times the other months
    select_aprils = list(range(3, 12*(2015-2000), 12))
    GDELT_masked_months = np.array(select_aprils)
    coordinates = np.meshgrid(I_range, I_range, A_range, GDELT_masked_months, np.ones(1))
    flattened_indices = [np.ravel(coords) for coords in coordinates]
    flattened_indices = np.vstack(flattened_indices)
    flattened_indices = flattened_indices.astype(np.int64)
    GDELT_mask = sparse.COO(coords=flattened_indices, data=np.ones(flattened_indices.shape[1]), shape=Y.shape)

    Y_mask = ICEWS_mask + TERRIER_mask + GDELT_mask
    Y_mask = Y_mask.todense()
    Y_mask[np.eye(Y_mask.shape[0]).astype(bool)] = 1
    Y_mask = sparse.COO(Y_mask.astype(np.int64))
    check_mask(Y_mask)
else:
    Y_mask = None

bptf_5mode = BPTF(data_shape=Y.shape, n_components=n_components)
filepath = f'bptf_5mode_{max_iter}iter_{n_components}_components_2000_2024.pkl'
if not os.path.exists(filepath):
    print('Fitting with BPTF')
    bptf_5mode.fit(Y, max_iter = max_iter, mask=Y_mask,verbose=False, missing_val=1)
    with open(filepath, 'wb') as f:
        pickle.dump(bptf_5mode, f)
else:
    print(f'{filepath} exists')
    with open(filepath, 'rb') as f:
        bptf_5mode = pickle.load(f)

# check the shapes
for j in range(len(Y.shape)):
    assert bptf_5mode.G_DK_M[j].shape == (Y.shape[j], n_components)

del bptf_5mode
gc.collect()

# Tensor factorisation with deterministic method ==================================================
filepath = 'nntf_parafac_5mode_2000_2018.pkl'
if os.path.exists(filepath):
    print('File exists')
else:
    print('Fitting with deterministic algorithm')
    cp_init = tensorly.cp_tensor.CPTensor(
        tensorly.decomposition._cp.initialize_cp(
            Y_2000_2018, non_negative = True, init = 'random', rank = n_components
            )
        )
    tensor_mu, _ = tensorly.decomposition.non_negative_parafac(
        Y_2000_2018, rank=n_components, init=cp_init, return_errors=True,
        verbose = 2
        )
    
    with open(filepath, 'wb') as f:
        pickle.dump(tensor_mu, f)