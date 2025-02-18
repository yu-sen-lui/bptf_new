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
n_components = 100

if os.path.exists('bptf_5mode_100iter.pkl'):
    with open('bptf_5mode_100iter.pkl', 'rb') as f:
        bptf_5mode = pickle.load(f)
    print('File exists')
else:
    bptf_5mode = BPTF(data_shape=Y.shape, n_components=n_components)
    bptf_5mode.fit(Y, max_iter = 100, verbose=True)
    print('Fitting with BPTF')
    with open('bptf_5mode_100iter.pkl', 'wb') as f:
        pickle.dump(bptf_5mode, f)

# check the shapes
for j in range(len(Y.shape)):
    assert bptf_5mode.G_DK_M[j].shape == (Y.shape[j], n_components)

del bptf_5mode
gc.collect()

# Tensor factorisation with deterministic method ==================================================
if os.path.exists('nntf_parafac_5mode.pkl'):
    with open('nntf_parafac_5mode.pkl', 'rb') as f:
        tensor_mu = pickle.load(f)
    print('File exists')
else:
    print('Fitting with deterministic algorithm')
    cp_init = tensorly.cp_tensor.CPTensor(
        tensorly.decomposition._cp.initialize_cp(
            Y, non_negative = True, init = 'random', rank = n_components
            )
        )
    tensor_mu, _ = tensorly.decomposition.non_negative_parafac(
        Y, rank=n_components, init=cp_init, return_errors=True
        )
    
    with open('nntf_parafac_5mode.pkl', 'wb') as f:
        pickle.dump(tensor_mu, f)