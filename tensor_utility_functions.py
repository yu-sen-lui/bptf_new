from tensorly.tenalg.core_tenalg.n_mode_product import multi_mode_dot
from tensorly.tenalg.core_tenalg._khatri_rao    import khatri_rao
from tensorly import backend as T
from tensorly.base import unfold
from tensorly.base import fold
import torch

def unfolding_dot_khatri_rao_memory(tensor, cp_tensor, mode):
    """mode-n unfolding times khatri-rao product of factors
    From https://tensorly.org/dev/_modules/tensorly/tenalg/core_tenalg/mttkrp.html

    Parameters
    ----------
    tensor : tl.tensor
        tensor to unfold
    factors : tl.tensor list
        list of matrices of which to the khatri-rao product
    mode : int
        mode on which to unfold `tensor`

    Returns
    -------
    mttkrp
        dot(unfold(tensor, mode), khatri-rao(factors))

    Notes
    -----
    Implemented as a sequence of Tensor-times-vectors products between a tensor
    and a Khatri-Rao product. The Khatri-Rao product is never computed explicitly,
    rather each column in the Khatri-Rao product is contracted with the tensor. This
    operation is implemented in Python and without making of use of parallelism, and it
    is therefore in general slower than the naive MTTKRP product.
    When the CP-rank of the CP-tensor is comparable to, or larger than,
    the dimensions of the input tensor, this method however requires much less
    memory.

    This method can also be implemented by taking n-mode-product with the full factors
    (faster but more memory consuming)::

        projected = multi_mode_dot(tensor, factors, skip=mode, transpose=True)
        ndims = T.ndim(tensor)
        res = []
        for i in range(factors[0].shape[1]):
            index = tuple([slice(None) if k == mode  else i for k in range(ndims)])
            res.append(projected[index])
        return T.stack(res, axis=-1)
    """
    mttkrp_parts = []
    weights, factors = cp_tensor
    rank = T.shape(factors[0])[1]
    for r in range(rank):
        component = multi_mode_dot(
            tensor, [T.conj(f[:, r]) for f in factors], skip=mode
        )
        mttkrp_parts.append(component)

    if weights is None:
        return T.stack(mttkrp_parts, axis=1)
    else:
        return T.stack(mttkrp_parts, axis=1) * T.reshape(weights, (1, -1))
    
def cp_to_tensor(cp_tensor):
    """
    Reconstructs the full tensor using the memory efficient Khatri-Rao product from tensorly
    Args:
        cp_tensor: tuple with (weights, factors)
        weights: list of CP weights
        factors: list of factor matrices of size (mode dim, rank)
    Returns:
        X_hat: the reconstructed tensor
    """
    weights, factors = cp_tensor
    shape = [factor_matrix.shape[0] for factor_matrix in factors]
    shape = tuple(shape)

    device = 'cpu' if factors[0].get_device() == -1 else 'cuda'
    mask = torch.ones(shape, dtype=torch.float64, device = device)

    X_0 = unfolding_dot_khatri_rao_memory(
        tensor=mask,
        cp_tensor=(weights, factors),
        mode=0
    )

    return fold(X_0, 0, shape)