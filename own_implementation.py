import numpy as np
import torch
import numpy.random as rn
import time
from tqdm import tqdm
import tensorly as tl
tl.set_backend('pytorch')
from sklearn.base import BaseEstimator, TransformerMixin
from tensor_utility_functions import unfolding_dot_khatri_rao_memory as unfolding_dot_khatri_rao
from tensor_utility_functions import kahan_diff, is_binary, KahanSum
from tensorly.cp_tensor import cp_to_tensor

import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)

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
        self.n_components = n_components

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
        self.epsilon = 0

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
        # Y_recon = tl.cp_tensor.cp_to_tensor(cp_tensor=(torch.ones(self.K, device=self.device), factors))
        Y_recon = cp_to_tensor(cp_tensor=(torch.ones(self.K, device=self.device), factors))

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
        Checks if the variational parameters and statistics are finite and not missing
        """
        assert torch.isfinite(self.shp_DK_M[m]).all().item(), "Infinite"
        assert torch.isfinite(self.rte_DK_M[m]).all().item(), "Infinite"
        assert torch.isfinite(self.E_DK_M[m]).all().item(), "Infinite"
        assert torch.isfinite(self.G_DK_M[m]).all().item(), "Infinite"
        assert torch.isfinite(self.Epsilon_DK_M[m]).all().item(), "Infinite"

        assert not torch.isnan(self.shp_DK_M[m]).any(), "NaN found in shp_DK_M"
        assert not torch.isnan(self.rte_DK_M[m]).any(), "NaN found in rte_DK_M"
        assert not torch.isnan(self.E_DK_M[m]).any(), "NaN found in E_DK_M"
        assert not torch.isnan(self.G_DK_M[m]).any(), "NaN found in G_DK_M"
        assert not torch.isnan(self.Epsilon_DK_M[m]).any(), "NaN found in Epsilon_DK_M"

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

    def _update_posterior_multinomial(self, m, data, mask=None):
        """
        Updates the posterior multinomial sufficient statistics for the latent sources
        """
        # \sum_{(m)} Mean along mode m for Poisson latent sources
        data = data if mask is None else data * mask
        data_hat = cp_to_tensor(cp_tensor=(None, self.G_DK_M))
        # data_hat = torch.clamp(data_hat, min=self.epsilon)
        self.Epsilon_DK_M[m] = self.G_DK_M[m] * unfolding_dot_khatri_rao(
            data / data_hat,
            (None, self.G_DK_M),
            m
        )

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
        self.rte_DK_M[m] = self.alpha * self.beta_M[m] + unfolding_dot_khatri_rao(
            mask,
            (None, self.E_DK_M),
            m
        )
        # self.rte_DK_M[m] = self.rte_DK_M[m].clamp(min=self.epsilon)

        # self.shp_DK_M[m] = self.shp_DK_M[m].clamp_(max=1e10)
        # self.rte_DK_M[m] = self.rte_DK_M[m].clamp_(min=self.epsilon, max=1e10)

    def _update_cache(self, m, data, mask = None):
        """
        Updates statistics required in updating variational and hyperparameters
        Args:
            m: mth mode of tensor. Natural number
            data: torch tensor,
            mask: binary torch tensor of the same dimensions as data. 1 is for observed data, 0 for unobserved
        """
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
        # data_hat = tl.cp_tensor.cp_to_tensor(cp_tensor=(None, self.G_DK_M))
        data_hat = cp_to_tensor(cp_tensor=(None, self.G_DK_M))
        self.Epsilon_DK_M[m] = self.G_DK_M[m] * unfolding_dot_khatri_rao(
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

        self._update_posterior_multinomial(m, data, mask)
        self._update_variational_params(m, data, mask)
        self._update_cache(m, data, mask)

        progressbar = tqdm(range(max_iter)) if verbose else range(max_iter)
        # check for negative elbo change
        neg_delta_list = []
        neg_delta_when = []
        elbo_list = [curr_elbo.item()]
        for itn in progressbar:

            s = time.time()

            # curr_elbo = self._elbo(data, mask)
            # print('test')

            # for m in modes:
            #     self._update_posterior_multinomial(m, data, mask)

            for m in modes:
                self._update_posterior_multinomial(m, data, mask)
                self._update_variational_params(m, data, mask)
                self._update_cache(m, data, mask)
                self._update_beta(m)
                self._check_mode(m)
            # for m in modes:
            #     self._update_beta(m)

            bound = self._elbo(data, mask)
            # delta = (bound - curr_elbo) / abs(curr_elbo)
            delta = kahan_diff(bound, curr_elbo) / abs(curr_elbo)
            
            if verbose:
                mem_alloc = torch.cuda.memory_allocated(0) / 1e9 if self.device == 'cuda' else None
                e = time.time() - s
                if mem_alloc is None:
                    progressbar.set_description(f'ELBO = {bound: .3f}, change = {delta: .3}, time taken = {e: .3}')
                else:
                    progressbar.set_description(f'ELBO = {bound: .3f}, change = {delta: .3}, time taken = {e: .3}, mem alloc = {mem_alloc: .3}GB')

            # check if the change is in the wrong direction
            assert delta >= 0.0, f"ELBO is negative: {delta}"
            elbo_list.append(bound.item())
            if delta < 0.0:
                neg_delta_list.append(delta.item())
                neg_delta_when.append(itn)
            curr_elbo = bound
            if abs(delta) < kwargs.get('tol', 1e-4):
                if verbose:
                    progressbar.set_description('Change is small enough, early break')
                break
        # print(f'Number of negative deltas: {len(neg_delta_list)}')
        # print(f'When do they occur? {neg_delta_when}')
        # print(f'what is their magnitude? {neg_delta_list}')
        # print(f'List of elbos: {elbo_list}')
        plt.plot(list(range(len(elbo_list))), elbo_list)
        plt.xlabel('Iter')
        plt.ylabel('Variational bound')
        plt.yscale('log')
        plt.show()

    def _gamma_bound_term_torch(self, pa, pb, qa, qb, compute_constant=False):
        """
        Computes the part of the variational bound that has the gamma distribution and gamma function terms
        Args:
            pa, pb: scalars or tensors. The prior hyperparameters
            qa, qb: tensors. Variational parameters
        Returns:
            out: the sum of gamma terms
        """
        out = torch.lgamma(qa) \
            - pa * torch.log(qb) \
            + (pa - qa) * torch.digamma(qa) \
            + qa * (1 - pb / qb)
        if compute_constant:
            out += pa * torch.log(pb) - torch.lgamma(pa)
        return out.to(self.device)

    def _elbo(self, data, mask=None):
        """
        Computes the variational lower bound
        Terms that don't change over iterations are omitted
        """
        # variational_bound = tl.cp_tensor.cp_to_tensor(cp_tensor=(None, self.E_DK_M), mask=None)
        # variational_bound = variational_bound if mask is None else variational_bound * mask
        # variational_bound = -variational_bound.sum()

        # for m in range(self.n_modes):
        #     variational_bound += ((self.alpha + self.Epsilon_DK_M[m] - 1.) * torch.log(self.G_DK_M[m])).sum()
        #     variational_bound += (-self.alpha*self.beta_M[m] * self.E_DK_M[m] + self.alpha * torch.log(self.beta_M[m])).sum()
        #     variational_bound -= ((self.shp_DK_M[m] - 1)*self.E_DK_M[m] - self.rte_DK_M[m]*self.E_DK_M[m] + 
        #                           self.shp_DK_M[m]*torch.log(self.rte_DK_M[m]) - torch.lgamma(self.shp_DK_M[m])).sum()
        
        # data = data if mask is None else data * mask
        # ratio = tl.cp_tensor.cp_to_tensor(cp_tensor=(None, self.G_DK_M), mask=None)
        # ratio /= ratio.sum(dim=-1, keepdim=True).clamp_min(self.epsilon)
        # pos_multinomial_prob = data * ratio
        # variational_bound -= (pos_multinomial_prob * torch.log(pos_multinomial_prob)).sum()

        # return variational_bound
        
        # bound = []

        # no_mask = True if mask is None else False
        mask = torch.ones_like(data, dtype=torch.float64, device=self.device) if mask is None else mask
        uttkrp_DK =  unfolding_dot_khatri_rao(
            mask,
            (None, self.E_DK_M),
            0
        )
        uttkrp_K = (self.E_DK_M[0] * uttkrp_DK).sum(dim=0)
        bound = -uttkrp_K.sum()
        # print(bound)
        # bound.append(-uttkrp_K.sum(dtype=torch.float64))
        assert torch.isfinite(uttkrp_K.sum()), "`bound` became NaN or Inf at first part"

        # old method for second part
        # data_recon = cp_to_tensor((None, self.G_DK_M))
        # data_recon = data_recon if mask is None else data_recon * mask
        # bound += (data * torch.log(
        #     data_recon.clamp(min=self.epsilon)
        # )).sum()

        # data_recon = tl.cp_tensor.cp_to_tensor((None, self.G_DK_M))
        # data_recon = cp_to_tensor((None, self.G_DK_M))
        for m in range(self.n_modes):
            assert (self.G_DK_M[m] > 0).all(), "Geometric mean is negative"
            assert torch.isfinite(self.G_DK_M[m]).all(), "Geometric mean blew up"
        # assert (data_recon > 0).all(), "recon contains zeros"
        # assert torch.isfinite(data_recon).all(), "recon blew up"

        # From our earlier printouts, we know that the geomexp is OK. It's not infinite or NaN
        # But data recon is, i.e. Y_hat
        # So, what we need to do is do log(theta_1) + \dots + log(theta_d)
        # then we multiply with y
        # get the coords of the nonzeros, don't form the data recon tensor, and compute the logged values directly

        # data_recon = data_recon if mask is None else data_recon * mask
        # this part is a computational bottleneck
        # runtime jumps from about 10s to 30s per iter
        # obs_coords = mask.to(torch.bool).cpu()
        # if no_mask:
        #     log_data_recon = torch.log(data_recon.clamp(min=self.epsilon))
        # else:
        #     data_recon = data_recon.cpu()
        #     data_recon = data_recon[obs_coords].to(self.device).clamp(min=self.epsilon)
        #     log_data_recon = torch.log(data_recon)

        #     data = data.cpu()
        #     data = data[obs_coords].to(self.device)
        # bound += (data * 
        #           log_data_recon
        # ).sum()
        # if no_mask:
        #     log_data_recon = torch.log(data_recon.clamp(min=self.epsilon))
        # else:
        #     obs_coords = mask.to(torch.bool)
        #     data = torch.masked_select(data, obs_coords)
        #     data_recon = torch.masked_select(data_recon, obs_coords).clamp(min=self.epsilon)
        #     log_data_recon = torch.log(data_recon)
        # bound += (data * log_data_recon).sum()
        # else:
        #     batch_size = 100000
        #     coords = (mask == 1).nonzero(as_tuple=False)
        #     for s in range(0, coords.size(0), batch_size):
        #         sub = coords[s:s+batch_size]
        #         data_batch = data[tuple(sub.T)]
        #         data_recon_batch = data_recon[tuple(sub.T)].clamp(min=self.epsilon)
        #         log_data_recon = torch.log(data_recon_batch)
        #         bound += (data_batch * log_data_recon).sum()

        # if no_mask:
        #     coords = (data != 0).nonzero(as_tuple=False)
        # else:
        #     coords = ((mask == 1) & (data != 0)).nonzero(as_tuple=False)
        coords = (data != 0).nonzero(as_tuple=False)
        batch_size = 100000
        log_G_DK_M = [torch.log(factor_matrix) for factor_matrix in self.G_DK_M]
        for s in range(0, coords.size(0), batch_size):
            sub = coords[s:s+batch_size]
            data_batch = data[tuple(sub.T)]
            
            M = self.n_modes # num_factors
            # K = self.K # num_components/rank
            log_data_recon_batch = log_G_DK_M[0][sub[:, 0], :]
            for m in range(1, M):
                log_data_recon_batch += log_G_DK_M[m][sub[:, m], :]
            
            log_data_recon_batch = torch.logsumexp(log_data_recon_batch, dim=1)

            # data_recon_batch = data_recon[tuple(sub.T)].clamp(min=self.epsilon)
            # log_data_recon = torch.log(data_recon_batch)
            bound += (data_batch * log_data_recon_batch).sum()
            # bound.append((data_batch * log_data_recon).sum(dtype=torch.float64))
        # print(bound)
        assert torch.isfinite(bound), "`bound` became NaN or Inf at second part"
        
        for m in range(self.n_modes):
            bound += self._gamma_bound_term_torch(pa=self.alpha,
                                                  pb=self.alpha * self.beta_M[m],
                                                  qa=self.shp_DK_M[m],
                                                  qb=self.rte_DK_M[m],
                                                  compute_constant=False).sum()

            bound += self.K \
                * self.data_shape[m] \
                * self.alpha.item() \
                * torch.log(self.beta_M[m].clamp(min=self.epsilon))
            # bound.append(
            #     self._gamma_bound_term_torch(pa=self.alpha,
            #                                       pb=self.alpha * self.beta_M[m],
            #                                       qa=self.shp_DK_M[m],
            #                                       qb=self.rte_DK_M[m],
            #                                       compute_constant=True).sum(dtype=torch.float64)
            # )
            # bound.append(
            #     self.K \
            #     * self.data_shape[m] \
            #     * self.alpha.item() \
            #     * torch.log(self.beta_M[m].clamp(min=self.epsilon))
            # )
        # print(bound)
        assert torch.isfinite(bound), "`bound` became NaN or Inf at third part"

        # bound = KahanSum(bound)
        
        return bound
    
    def fit(self, data, mask=None, **kwargs):
        """
        Call this to fit the model to the given data and mask
        """
        assert data.shape == self.data_shape, f"Expected shape {self.data_shape} but got {data.shape}"
        if mask is not None:
            assert is_binary(mask), 'Mask is not binary'

        self._init()
        self._update(data, mask, None, **kwargs)
        return self