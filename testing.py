# packages ========================================================================================
import sys
from IPython.display import display, Javascript

def restart_kernel():
    """Restart the Jupyter Notebook kernel to reflect changes in modules and packages."""
    display(Javascript("Jupyter.notebook.kernel.restart()"))
    print("Kernel is restarting...")

restart_kernel()

import bptf
# from bptf import BPTF
from own_implementation import BPTF
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

assert os.path.exists('sptensor.pkl'), 'No such file.'
with open('sptensor.pkl', 'rb') as f:
    Y = pickle.load(f)

device = 'cuda'

Y = Y.todense()
Y = Y[:, :, :, :, 0]

# mask
mask = np.ones(Y.shape)
diagonal_indices = np.arange(0, Y.shape[0])
mask[diagonal_indices, diagonal_indices, :, :] = 0
mask = mask.astype(np.int64)
mask = torch.tensor(mask, dtype=torch.float64, device=device)

# I_range = np.arange(Y.shape[0])
# A_range = np.arange(Y.shape[2])
# T_range = np.arange(Y.shape[3])

# I, A, T = np.meshgrid(I_range, A_range, T_range, indexing='ij')
# I_flattened = np.ravel(I)
# A_flattened = np.ravel(A)
# T_flattened = np.ravel(T)

# mask_coords = np.vstack([I_flattened, I_flattened, A_flattened, T_flattened])
# mask = sparse.COO(coords=mask_coords, data=np.ones(shape = mask_coords.shape[1]), shape=Y.shape)

Y = torch.tensor(Y, dtype=torch.float64, device=device)

model = BPTF(data_shape=Y.shape, n_components=100)
model.fit(Y, max_iter = 50, mask=mask,verbose=True, tol=1e-8)

with open('bptf_icews.pkl', 'wb') as f:
    pickle.dump(model, f)
