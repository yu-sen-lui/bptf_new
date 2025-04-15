import numpy as np
import torch
import numpy.random as rn
import time
from tqdm import tqdm
import tensorly as tl
tl.set_backend('pytorch')
from sklearn.base import BaseEstimator, TransformerMixin

# switching to memory efficient mttkrp
# from tensorly.tenalg.core_tenalg.mttkrp import unfolding_dot_khatri_rao_memory
# tl.tenalg.register_backend_method("unfolding_dot_khatri_rao", unfolding_dot_khatri_rao_memory)
# tl.tenalg.use_dynamic_dispatch()

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
        self.shp_DK_M = [torch.ones((D, self.K), dtype=torch.float64, device=self.device) for D in self.data_shape]
        self.rte_DK_M = [torch.ones((D, self.K), dtype=torch.float64, device=self.device) for D in self.data_shape]

        # sufficient statistics of latent source posterior distribution
        self.Epsilon_DK_M = [torch.ones((D, self.K), dtype=torch.float64, device=self.device) for D in self.data_shape]

        # arithmetic and geometric expectation of factor matrices
        self.E_DK_M = [torch.ones((D, self.K), dtype=torch.float64, device=self.device) for D in self.data_shape]
        self.G_DK_M = [torch.ones((D, self.K), dtype=torch.float64, device=self.device) for D in self.data_shape]

        # small positive number to prevent division by 0
        self.epsilon = 10e-10

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
        Y_recon = tl.cp_tensor.cp_to_tensor(cp_tensor=(torch.ones(self.K, device=self.device), factors))

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
        Returns:
            instance objects listed above
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
        self.E_DK_M[m] = self.shp_DK_M[m] / self.rte_DK_M[m]
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

    def _init(self, modes = None, **kwargs):
        """
        Initialise:
        Variational parameters
        Sufficient statistics of surrogate distribution
        Prior rate hyperparameter
        Args:
            m: int. mth mode of the tensor
        """

        modes = range(self.n_modes) if modes is None else list(set(modes))

        for m in modes:
            self._init_mode(m, **kwargs)

    def _update_variational_params(self, m, data, mask=None):
        """
        Does the CAVI update for a single mode
        Also updates the variational surrogate distribution statistics
        Args:
            m: mth mode of the data tensor
            data: data tensor, torch tensor with same device as self.device
            mask: binary tensor with same shape as data tensor, 1 for observed
        """
        if mask is None:
            mask = torch.ones(self.data_shape, device=self.device, dtype=torch.float64)
        
        # update variational shape parameter
        # equation 4
        self.shp_DK_M[m] = self.alpha + self.Epsilon_DK_M[m]
        # equation 5
        self.rte_DK_M[m] = self.alpha * self.beta_M[m] + tl.tenalg.unfolding_dot_khatri_rao(
            mask,
            (torch.ones(self.K, device=self.device, dtype=torch.float64), self.E_DK_M),
            m
        )

        self.shp_DK_M[m].clamp_(max=1e10)
        self.rte_DK_M[m].clamp_(max=1e10)

    def _update_cache(self, m, data, mask = None):
        """
        Updates statistics required in updating variational and hyperparameters
        Args:
            m: mth mode of tensor. Natural number
            data: torch tensor,
            mask: binary torch tensor of the same dimensions as data. 1 is for observed data, 0 for unobserved
        """
        # \sum_{(m)} Mean along mode m for Poisson latent sources
        data = data if mask is None else data * mask
        data_hat = tl.cp_tensor.cp_to_tensor(cp_tensor=(None, self.G_DK_M))
        self.Epsilon_DK_M[m] = self.G_DK_M[m] * tl.tenalg.unfolding_dot_khatri_rao(
            data / data_hat + self.epsilon,
            (torch.ones(self.K, device=self.device, dtype=torch.float64), self.G_DK_M),
            m
        )
        self.G_DK_M[m] = torch.exp(torch.digamma(self.shp_DK_M[m]) - torch.log(self.rte_DK_M[m]))
        self.E_DK_M[m] = self.shp_DK_M[m] / self.rte_DK_M[m]

    def _update_beta(self, m):
        """
        Updates prior rate hyperparameter using empirical Bayes approach
        Args:
            m: integer, mth mode of the data tensor
        """
        self.beta_M[m] = 1. / torch.mean(self.E_DK_M[m])

    def _clamp_component(self, data, m, mask=None):
        """
        Sets this mode's cached variables to constants
        """
        self.E_DK_M[m] = self.G_DK_M[m]
        self.beta_M[m] = 1. / torch.mean(self.E_DK_M[m])
        data = data if mask is None else data * mask
        data_hat = tl.cp_tensor.cp_to_tensor(cp_tensor=(None, self.G_DK_M))
        self.Epsilon_DK_M[m] = self.G_DK_M[m] * tl.tenalg.unfolding_dot_khatri_rao(
            data / data_hat + self.epsilon,
            (torch.ones(self.K, device=self.device, dtype=torch.float64), self.G_DK_M),
            m
        )

    def _update(self, data, mask=None, modes=None, **kwargs):
        """
        Call this to run the update for max_iter times
        """
        modes = range(self.n_modes) if modes is None else list(set(modes))

        for m in range(self.n_modes):
            if m not in modes:
                self._clamp_component(data, m, mask=mask)

        curr_elbo = self._elbo(data, mask)
        
        verbose = kwargs.get('verbose', True)
        max_iter = kwargs.get('max_iter', 100)

        progressbar = tqdm(range(max_iter)) if verbose else range(max_iter)
        for itn in progressbar:

            s = time.time()
            for m in modes:
                self._update_variational_params(m, data, mask)
                self._update_cache(m, data, mask)
                self._update_beta(m)
                self._check_mode(m)
            bound = self._elbo(data, mask)
            delta = (bound - curr_elbo) / abs(curr_elbo)
            if verbose:
                e = time.time() - s
                progressbar.set_description(f'ELBO = {bound}, change = {delta}, time taken = {e}')

            # check if the change is in the wrong direction
            assert delta > -1e-8, f"ELBO is negative: {delta}"
            curr_elbo = bound
            if abs(delta) < kwargs.get('tol', 1e-4):
                if verbose:
                    progressbar.set_description('Change is small enough, early break')
                break

    def _elbo(self, data, mask=None):
        """
        Computes the variational lower bound
        Terms that don't change over iterations are omitted
        """
        variational_bound = tl.cp_tensor.cp_to_tensor(cp_tensor=(None, self.E_DK_M), mask=None)
        variational_bound = variational_bound if mask is None else variational_bound * mask
        variational_bound = -variational_bound.sum()

        for m in range(self.n_modes):
            variational_bound += ((self.alpha + self.Epsilon_DK_M[m] - 1.) * torch.log(self.G_DK_M[m])).sum()
            variational_bound += (-self.alpha*self.beta_M[m] * self.E_DK_M[m] + self.alpha * torch.log(self.beta_M[m])).sum()
            variational_bound -= ((self.shp_DK_M[m] - 1)*self.E_DK_M[m] - self.rte_DK_M[m]*self.E_DK_M[m] + 
                                  self.shp_DK_M[m]*torch.log(self.rte_DK_M[m]) - torch.lgamma(self.shp_DK_M[m])).sum()
        
        data = data if mask is None else data * mask
        ratio = tl.cp_tensor.cp_to_tensor(cp_tensor=(None, self.G_DK_M), mask=None)
        ratio /= ratio.sum(dim=-1, keepdim=True).clamp_min(self.epsilon)
        pos_multinomial_prob = data * ratio
        variational_bound -= (pos_multinomial_prob * torch.log(pos_multinomial_prob)).sum()

        return variational_bound
    
    def fit(self, data, mask=None, **kwargs):
        """
        Call this to fit the model to the given data and mask
        """
        assert data.shape == self.data_shape, f"Expected shape {self.data_shape} but got {data.shape}"

        self._init()
        self._update(data, mask, None, **kwargs)
        return self