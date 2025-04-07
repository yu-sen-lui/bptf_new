import numpy as np
import torch
import numpy.random as rn
import time
from tqdm import tqdm
import tensorly as tl
tl.set_backend('pytorch')
from sklearn.base import BaseEstimator, TransformerMixin

class BPTF(BaseEstimator, TransformerMixin):
    def __init__(self, data_shape, n_components, alpha = 0.1, device="cpu"):
        """
        BPTF object to fit with
        Args:
            data_shape: tuple of int you get from .shape attribute
            n_components: natural number, number of components or dimension of factor matrices
            alpha: positive float. Reflects the sparsity of the tensor. \alpha << 1 means sparse tensor
        """
        assert isinstance(n_components, int) and n_components > 0, 'Number of components is a natural number'
        assert alpha > 0, 'Shape parameter for gamma prior must be positive'

        self.device = device

        self.data_shape = data_shape
        self.n_modes = len(data_shape)
        self.K = n_components

        # hyperparameters
        self.beta_M = torch.ones(self.n_modes, dtype=torch.float64, device=device)
        self.alpha = torch.ones(1, dtype=torch.float64, device=device) * alpha

        # variational parameters
        # distributions are gamma with shape and RATE parameters
        self.shp_DK_M = [torch.ones(D, self.K, dtype=torch.float64, device=device) for D in self.data_shape]
        self.rte_DK_M = [torch.ones(D, self.K, dtype=torch.float64, device=device) for D in self.data_shape]

        # arithmetic and geometric expectation of factor matrices
        self.E_DK_M = np.array([np.ones((D, self.K)) for D in self.data_shape], dtype='object')  
        self.G_DK_M = np.array([np.ones((D, self.K)) for D in self.data_shape], dtype='object')

    def reconstruct(self, mask=None, drop_diag=False, fill_value = 0, style='arithmetic'):
        """
        Reconstructs the tensor using the factor matrices
        Takes the mean of the surrogate distributions of the factor matrices and uses those as factor matrix estimates
        Args:
            mask: binary tensor of the same shape as the original tensor, 1 where the element is observed, 0 otherwise
            drop_diag: Boolean. Set to true to replace diagonal of tensor with fill_value
            fill_value: float
            style: arithmetic to take the arithmetic mena of the surrogate distributions, and geometric for the geometric mean
        """
        assert style in ['arithmetic', 'geometric'], "Wrong style"
        
        factors = self.G_DK_M if style == 'geometric' else self.E_DK_M
        Y_recon = tl.cp_tensor.cp_to_tensor(cp_tensor=(torch.ones(self.K, factors)))

        # fill in mask with fill_value because they are not observed
        if mask is not None:
            assert Y_recon.shape == mask.shape, f"Mask shape of {mask.shape} does not match that of the tensor {Y_recon.shape}"
            Y_recon[mask == 0] = fill_value

        # fill in diagonals with fill_value
        if drop_diag:
            assert Y_recon.shape[0] == Y_recon.shape[1], "First 2 dimensions should match"
            diagonals = torch.diagonal(Y_recon, offset=0, dim1=0, dim2=1)
            diagonals.fill_(fill_value)

        return Y_recon
    
    def _init_mode(self, m, **kwargs):
        """
        Helper function to initialise:
        Variational parameters
        Sufficient statistics of surrogate distribution
        Prior rate hyperparameter
        Args:
            m: int. mth mode of the tensor
        """
        # default initial shape and rate for initialising variational parameters
        shp = kwargs.get("init_shp", 100.); rte = kwargs.get("init_rte", 1.)
        D = self.data_shape[m]
        
        # initialising variational parameters
        concentration = torch.tensor(shp, device=self.device, dtype=torch.float64)
        rate_tensor = torch.tensor(1. / rte, device=self.device, dtype=torch.float64)
        gamma_dist = torch.distributions.Gamma(concentration, rate_tensor)
        self.shp_DK_M[m] = gamma_dist.sample((D, self.K))
        self.rte_DK_M[m] = gamma_dist.sample((D, self.K))

        # initialising statistics of variational surrogate distributions
        self.E_DK_M[m] = 1. / self.shp_DK_M[m] / self.rte_DK_M[m]
        self.G_DK_M[m] = torch.exp(torch.digamma(self.shp_DK_M[m]) - torch.log(self.rte_DK_M[m]))

        # initialise prior rate hyperparameter using empirical bayes method
        self.beta_M[m] = 1. / self.E_DK_M[m].mean()

        self._check_mode(m)

    def _check_mode(self, m):
        """
        Checks if the variational parameters and statistics are finite
        """
        assert torch.isfinite(self.shp_DK_M[m]).all().item(), "Infinite"
        assert torch.isfinite(self.rte_DK_M[m]).all().item(), "Infinite"
        assert torch.isfinite(self.E_DK_M[m]).all().item(), "Infinite"
        assert torch.isfinite(self.G_DK_M[m]).all().item(), "Infinite"

    def _init(self, **kwargs):
        """
        Initialise:
        Variational parameters
        Sufficient statistics of surrogate distribution
        Prior rate hyperparameter
        Args:
            m: int. mth mode of the tensor
        """

        modes = range(self.n_modes)

        for m in modes:
            self._init_mode(m, **kwargs)

    def _update_variational_params(self, m, data, mask=None):
        """
        Does the CAVI update for a single mode
        Also updates the variational surrogate distribution statistics
        Args:
            m: mth mode of the data tensor
            
        """