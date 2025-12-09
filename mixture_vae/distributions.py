import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as D
from typing import Any, Iterable
from abc import ABC, abstractmethod

from mixture_vae.utils import (
    check_nonbatch,
    safe_T,
    split_or_validate_features,
    get_dim,
    _to_tensor_dict,
)


class Distribution(ABC):
    """Base class to define distributions."""
    def __init__(self, ref_parameters: dict[str, Any]):
        self.ref_parameters = _to_tensor_dict(ref_parameters)
        self.n_params = len(self.ref_parameters.values())
        self.param_dims = [get_dim(val) for val in self.ref_parameters.values()]
        self.sample_dim = self.param_dims[0]
        # usually all parameters are in the sampling dimension,
        # can be overwritten for dist where its not the case

    def get_reference_params(self, z=None):
        """
        If z is a tensor, broadcast the reference parameters to its batch size,
        making sure we are on the same device / dtype as z.
        """
        if z is None:
            return self.ref_parameters
        else:
            assert isinstance(z, torch.Tensor) or isinstance(
                z, np.ndarray
            ), f"z must be a torch.Tensor or an np.ndarray but got {type(z)}"

            if isinstance(z, torch.Tensor):
                device = z.device
                dtype = z.dtype
                # stack params on same device/dtype as z
                param_tensors_list = [
                    val.to(device=device, dtype=dtype).contiguous().view(1, -1)
                    for _, val in self.ref_parameters.items()
                ]
                param_tensor = torch.cat(param_tensors_list, dim=1)
                batch_size = z.size(0)
                param_tensor = param_tensor.tile((batch_size, 1))
                return param_tensor
            else:
                # numpy path (no device issue, but fix contig/reshape)
                param_arr_list = [
                    np.asarray(val).reshape(1, -1)
                    for _, val in self.ref_parameters.items()
                ]
                param_arr = np.concatenate(param_arr_list, axis=1)
                batch_size = z.shape[0]
                param_arr = np.broadcast_to(
                    param_arr, (batch_size, param_arr.shape[1])
                )
                return param_arr


class UniformDistribution(Distribution):
    def __init__(self, ref_parameters):
        super().__init__(ref_parameters)
        self.parametric_kl = True
        self.n_params = 2  # [a,b]

    def get_ref_proba(self):
        a = self.ref_parameters["a"]
        b = self.ref_parameters["b"]
        d = self.param_dims[0]  # necessarily >= 1
        support_proba = 1 / (b - a + 1e-6)
        # to normalize across dimensions (the sum of probas across dims need to be 1)
        return support_proba / d

    def sample(self, latent_params, batch_size):
        a, b = split_or_validate_features(latent_params, self.param_dims)

        if check_nonbatch(a):
            # if its only one distribution we extend to batch_size
            shape = (batch_size, self.sample_dim)
        else:
            shape = tuple(list(latent_params.size())[:-1] + [self.sample_dim])
            if batch_size != shape[0]:
                print(
                    f"WARNING: Mismatch between param batch dim ({shape[0]}) and batch_size ({batch_size})!"
                )
                print(
                    f"Ignoring batch_size to prefer the first dimension of params dim: {shape[0]}"
                )
        # Reparametrization trick
        noise = torch.rand(shape, device=a.device, dtype=a.dtype)
        return (b - a) * noise + a

    def log_likelihood(self, x, params):
        a, b = split_or_validate_features(params, self.param_dims)
        eps = 1e-6

        # normalize devices and dtypes
        device = x.device
        dtype = x.dtype
        x = x.to(device=device, dtype=dtype)
        a = a.to(device=device, dtype=dtype)
        b = b.to(device=device, dtype=dtype)

        width = (b - a).clamp_min(eps)

        # if x, a, b are B×d: "inside" means inside on every dim
        inside = (x >= a) & (x <= b)
        if inside.dim() > 1:
            inside = inside.all(dim=1)  # B
            log_width_sum = torch.log(width).sum(dim=1)  # B (diagonal/indep dims)
        else:
            # 1D case: B
            log_width_sum = torch.log(width)

        logp = -log_width_sum
        # outside support: -inf
        neg_inf = torch.full_like(logp, float("-inf"))
        return torch.where(inside, logp, neg_inf)

    def kl_divergence(self, input_params, target_params):
        in_a, in_b = split_or_validate_features(input_params, self.param_dims)
        tg_a, tg_b = split_or_validate_features(target_params, self.param_dims)
        eps = 1e-6

        # put everything on the same device/dtype as in_a
        device = in_a.device
        dtype = in_a.dtype
        in_a = in_a.to(device=device, dtype=dtype)
        in_b = in_b.to(device=device, dtype=dtype)
        tg_a = tg_a.to(device=device, dtype=dtype)
        tg_b = tg_b.to(device=device, dtype=dtype)

        in_w = (in_b - in_a).clamp_min(eps)
        tg_w = (tg_b - tg_a).clamp_min(eps)

        # we need to check support for all dim:
        # [in_a, in_b] ⊆ [tg_a, tg_b] in all dims
        incl = (in_a >= tg_a) & (in_b <= tg_b)
        incl = incl.all(dim=1) if incl.dim() > 1 else incl

        # KL = sum_d log(tg_w / in_w)
        if in_w.dim() > 1:
            kl = torch.log(tg_w / in_w).sum(dim=1)
        else:
            kl = torch.log(tg_w / in_w)

        # where it's not on the support it's inf
        pos_inf = torch.full_like(kl, float("inf"))
        return torch.where(incl, kl, pos_inf)


class NormalDistribution(Distribution):
    def __init__(self, ref_parameters):
        super().__init__(ref_parameters)
        self.parametric_kl = True
        self.n_params = 2  # mu, sigma

    def constraints(self, params):
        constrained = params.clone()
        # softplus for strict positivity + epsilon
        # (to avoid killing the log-likelihood with 0 variance)
        constrained[:, 1] = F.softplus(params[:, 1]) + 1e-6
        return constrained

    def sample(self, latent_params, batch_size):
        mu, std = split_or_validate_features(latent_params, self.param_dims)
        if check_nonbatch(mu):
            # if its only one distribution we extend to batch_size
            shape = (batch_size, self.sample_dim)
        else:
            shape = tuple(list(latent_params.size())[:-1] + [self.sample_dim])
            if batch_size != shape[0]:
                print(
                    f"WARNING: Mismatch between param batch dim ({shape[0]}) and batch_size ({batch_size})!"
                )
                print(
                    f"Ignoring batch_size to prefer the first dimension of params dim: {shape[0]}"
                )
        # Reparametrization trick
        noise = torch.randn(shape, device=mu.device, dtype=mu.dtype)
        return mu + std * noise

    def log_likelihood(self, x, params):
        mu, std = split_or_validate_features(params, self.param_dims)
        eps = 1e-6

        # put everything on same device/dtype as x
        device = x.device
        dtype = x.dtype
        x = x.to(device=device, dtype=dtype)
        mu = mu.to(device=device, dtype=dtype)
        std = std.to(device=device, dtype=dtype)

        var = (std**2).clamp_min(eps)

        # feature dimension
        d = mu.size(1) if mu.dim() > 1 else 1

        # log-likelihood for diagonal Gaussian
        two_pi = torch.tensor(2.0 * torch.pi, device=device, dtype=dtype)
        log_norm = -0.5 * (
            d * torch.log(two_pi) + torch.log(var).sum(dim=1 if var.dim() > 1 else 0)
        )
        quad = -0.5 * (((x - mu) ** 2) / var).sum(dim=1 if var.dim() > 1 else 0)
        return log_norm + quad

    def kl_divergence(self, input_params, target_params):
        mu0, std0 = split_or_validate_features(input_params, self.param_dims)  # p
        mu1, std1 = split_or_validate_features(target_params, self.param_dims)  # q
        eps = 1e-6

        # Ensure all parameters are on the same device as the inputs
        device = mu0.device
        dtype = mu0.dtype

        mu0 = mu0.to(device=device, dtype=dtype)
        std0 = std0.to(device=device, dtype=dtype)
        mu1 = mu1.to(device=device, dtype=dtype)
        std1 = std1.to(device=device, dtype=dtype)

        var0 = (std0**2).clamp_min(eps)
        var1 = (std1**2).clamp_min(eps)

        # feature dimension
        k = mu0.size(1) if mu0.dim() > 1 else 1

        term1 = (var0 / var1).sum(dim=1 if var0.dim() > 1 else 0)
        term2 = (((mu1 - mu0) ** 2) / var1).sum(dim=1 if var1.dim() > 1 else 0)
        term3 = -k + (torch.log(var1) - torch.log(var0)).sum(
            dim=1 if var0.dim() > 1 else 0
        )

        return 0.5 * (term1 + term2 + term3)


class NegativeBinomial(Distribution):
    def __init__(self, ref_parameters):
        super().__init__(ref_parameters)
        self.parametric_kl = False
        self.n_params = 2 # logits, r
        # logits to avoid issues for loglikelihood when p hits 0 or 1 (-inf ll)
    
    def constraints(self, params):
        params = params.double()
        constrained = params.clone()
        # ensure positivity on counts
        r_dims = self.param_dims[1]
        logits_dims = self.param_dims[0]
        start, end = logits_dims, r_dims + logits_dims
        constrained[:,start:end] = F.softplus(params[:,start:end]) + 1e-6
        return constrained
    
    def sample(self, latent_params, batch_size):
        # the NB is only used to generate samples from decoder
        # output params and the reconstruction loss 
        # only relies on the parameters so no worries 
        # about using no reparam trick here 
        logits, r = split_or_validate_features(latent_params, 
                                          self.param_dims)
        NB = torch.distributions.NegativeBinomial(total_count=r, 
                                                logits=logits)
        return NB.sample(batch_size)
    
    def log_likelihood(self, x, params):
        # NB has a differentiable log likelihood
        logits, r = split_or_validate_features(params, 
                                          self.param_dims)
        NB = torch.distributions.NegativeBinomial(total_count=r, 
                                                logits=logits)
        return NB.log_prob(x)
    
    def kl_divergence(self, input_params, target_params):
        """
        No closed form. You need to compute KL using likelihoods directly.
        """
        pass

class Poisson(Distribution):
    def __init__(self, ref_parameters):
        super().__init__(ref_parameters)
        self.parametric_kl = False
        self.n_params = 1  # lambda

    def constraints(self, params):
        params = params.double()
        constrained = params.clone()
        # ensure > 0 on lambda
        constrained[:, 0] = F.softplus(params[:, 0]) + 1e-6
        return constrained

    def sample(self, latent_params, batch_size):
        # Same remark as for the NB
        lmbda = split_or_validate_features(latent_params, self.param_dims)
        P = torch.distributions.Poisson(rate=lmbda)
        return P.sample((batch_size,))

    def log_likelihood(self, x, params):
        # Poisson has a differentiable log likelihood
        lmbda = split_or_validate_features(params, self.param_dims)
        P = torch.distributions.Poisson(rate=lmbda)
        return P.log_prob(x)

    def kl_divergence(self, input_params, target_params):
        in_l = split_or_validate_features(input_params, self.param_dims)
        target_l = split_or_validate_features(target_params, self.param_dims)

        # normalize shapes + device/dtype
        device = in_l.device
        dtype = in_l.dtype
        in_l = in_l.contiguous().view(-1).to(device=device, dtype=dtype)
        target_l = target_l.contiguous().view(-1).to(device=device, dtype=dtype)

        return in_l * torch.log(1e-8 + in_l / (target_l + 1e-8)) + target_l - in_l

class Student(Distribution):
    def __init__(self, ref_parameters):
        super().__init__(ref_parameters)
        self.parametric_kl = False
        # Expects: df, mu, scale
        self.n_params = 3

    def constraints(self, params):
        """
        Enforce positivity on Degrees of Freedom (df) and Scale.
        Assumes order in ref_parameters: {df, mu, scale}
        """
        params = params.double()
        constrained = params.clone()
        
        # Dimensions
        dim_df = self.param_dims[0]
        dim_mu = self.param_dims[1]
        dim_scale = self.param_dims[2]

        # 1. df (index 0) must be > 0
        constrained[:, :dim_df] = F.softplus(params[:, :dim_df]) + 1e-6
        
        # 2. mu (index 1) is Real (no constraint needed)
        
        # 3. scale (index 2) must be > 0
        start_scale = dim_df + dim_mu
        constrained[:, start_scale:] = F.softplus(params[:, start_scale:]) + 1e-6
        
        return constrained

    def sample(self, latent_params, batch_size):
        # Split params
        df, mu, scale = split_or_validate_features(latent_params, self.param_dims)
        
        # PyTorch StudentT supports rsample (reparameterization trick)
        dist = D.StudentT(df=df, loc=mu, scale=scale)
        
        # If latent_params matches batch_size, we just sample once
        # If latent_params is a single reference (1, D), we expand to batch_size
        if df.size(0) == 1 and batch_size > 1:
             return dist.rsample((batch_size,))
        else:
             return dist.rsample()

    def log_likelihood(self, x, params):
        df, mu, scale = split_or_validate_features(params, self.param_dims)
        dist = D.StudentT(df=df, loc=mu, scale=scale)
        return dist.log_prob(x)

    def kl_divergence(self, input_params, target_params):
        """
        No closed form KL for Student-T. 
        The Model will fall back to Monte Carlo estimation (log q - log p).
        """
        pass