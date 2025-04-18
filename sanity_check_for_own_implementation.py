import numpy as np, torch, tensorly as tl
from own_implementation import BPTF as BPTF_torch
import pickle

with open('sptensor.pkl', 'rb') as f:
    Y = pickle.load(f)
Y = Y[:, :, :, :24, :]
shape = Y.shape
K = 10
print(Y.shape)

# mask_np    = (Y == 0).astype(int)      # 1 = missing for Aaron
# mask_torch = torch.tensor(1 - mask_np) # 1 = observed for ours

mask_np = np.zeros(shape, dtype=int)
mask_np[:, :, :, 3, :] = 1
mask_torch = torch.tensor(1 - mask_np) 

max_iter = 200
# ---------- Aaron's model (needs NumPy backend) ----------
tl.set_backend("numpy")
from bptf import BPTF as BPTF_np          # import *after* backend switch
model_np = BPTF_np(shape, K).fit(Y.todense(), None, verbose=False, max_iter=max_iter, tol=1e-10)
recon_np = model_np.reconstruct()

# ---------- Our PyTorch model ----------
tl.set_backend("pytorch")
device = 'cuda'
model_torch = BPTF_torch(shape, K, device=device).fit(
    torch.tensor(Y.todense(), dtype=torch.float64, device=device),
    mask=None,
    verbose=True, max_iter=max_iter, tol=1e-10)

# Compare reconstructions
recon_torch = model_torch.reconstruct().cpu().numpy()
print("MAE :", np.mean(np.abs(recon_np - recon_torch)))
print("Relative L1 error:", np.sum(np.abs(recon_np - recon_torch)) / np.sum(np.abs(recon_np)))