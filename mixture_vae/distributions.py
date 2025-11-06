import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as D
from typing import Any, Iterable
from abc import ABC, abstractmethod

from mixture_vae.utils import (check_nonbatch, 
                               safe_T, 
                               split_or_validate_features, 
                               get_dim,
                               _to_tensor_dict
                               )

class Distribution(ABC):
    """Base class to define distributions."""
    def __init__(self, ref_parameters: dict[str:Any]):
        self.ref_parameters = _to_tensor_dict(ref_parameters)
        self.n_params = len(self.ref_parameters.values())
        self.param_dims = [get_dim(val) 
                           for val in self.ref_parameters.values()]
        self.sample_dim = self.param_dims[0]
        # usually all parameters are in the sampling dimension, 
        # can be overwritten for dist where its not the case

    def get_reference_params(self, z=None):
        if z is None:
            return self.ref_parameters
        else:
            assert isinstance(z, torch.Tensor) or isinstance(z, np.ndarray), f"z must be a torch.Tensor or an np.ndarray but got {type(z)}"

            if isinstance(z, torch.Tensor):
                param_tensors_list = [val.contiguous().view(1,-1) 
                                      for _, val in self.ref_parameters.items()]
                param_tensor = torch.cat(param_tensors_list, axis=1)
                batch_size = z.size(0)
                param_tensor = param_tensor.tile((batch_size,1))
                return param_tensor
            else:
                param_arr_list = [np.array(val).contiguous().view(1,-1) 
                                    for _, val in self.ref_parameters.items()]
                param_arr = np.concat(param_arr_list, axis=1)
                batch_size = z.shape[0]
                param_arr = np.broadcast_to(param_arr, (batch_size,param_arr.shape[1]))
                return param_arr
    
    def constraints(self, params):
        """
        Method to apply constraints on predicted parameters
        By default no constraints are imposed.
        """
        return params

    @abstractmethod
    def sample(self, 
               latent_params: tuple[torch.Tensor], 
               batch_size: int):
        """
        Abstract method to define the sampling procedure.
        """
        pass

    @abstractmethod
    def log_likelihood(self, 
                       x: torch.Tensor, 
                       params: tuple[torch.Tensor]):
        """
        Abstract method to define how to compute the log likelihood
        Here params should be the parameters for ONE distribution !
        """
        pass

    @abstractmethod
    def kl_divergence(self, 
                      input_params: tuple[torch.Tensor], 
                      target_params: tuple[torch.Tensor]):
        """
        Abstract method to define how to compute the KL divergence
        with a parametric form. If the disttribution does not have 
        a closed form formula it should be left empty.
        """
        pass

class UniformDistribution(Distribution):
    def __init__(self, ref_parameters):
        super().__init__(ref_parameters)
        self.parametric_kl = True
        self.n_params = 2 # [a,b]
    
    def get_ref_proba(self):
        a = self.ref_parameters["a"]
        b = self.ref_parameters["b"]
        d = self.param_dims[0] # necessarily >= 1
        support_proba = 1 / (b - a + 1e-6)
        return support_proba / d # to normalize across dimensions (the sum of probas across dims need to be 1)
    
    def sample(self, latent_params, batch_size):
        a, b = split_or_validate_features(latent_params, 
                                          self.param_dims)
        
        if check_nonbatch(a):
            # if its only one distirbution we extend to batch_size
            shape = [batch_size, self.sample_dim]
            shape = tuple(shape)
        else:
            shape = [latent_params.size(0), self.sample_dim]
            shape = tuple(shape)
            if batch_size != shape[0]:
                print(f"WARNING: Mismatch between param batch dim ({shape[0]}) and batch_size ({batch_size})!")
                print(f"Ignoring batch_size to prefer the first dimension of params dim: {shape[0]}")
        # Reparametrization trick
        noise = torch.rand(shape, device=a.device, dtype=a.dtype)
        return (b - a) * noise + a
    
    def log_likelihood(self, x, params):
        a, b = split_or_validate_features(params, self.param_dims)
        eps = 1e-6

        # normalize dtypes
        x = x.double()
        a = a.double()
        b = b.double()

        width = (b - a).clamp_min(eps)

        # if x, a, b are B×d: "inside" means inside on every dim
        inside = (x >= a) & (x <= b)
        if inside.dim() > 1:
            inside = inside.all(dim=1)            # B
            log_width_sum = torch.log(width).sum(dim=1)  # B (diagonal/indep dims)
        else:
            # 1D case: B
            log_width_sum = torch.log(width)

        logp = -log_width_sum
        # outside support: -inf
        neg_inf = torch.full_like(logp, float('-inf'))
        return torch.where(inside, logp, neg_inf)
    
    def kl_divergence(self, input_params, target_params):
        in_a, in_b = split_or_validate_features(input_params, 
                                                self.param_dims)
        tg_a, tg_b = split_or_validate_features(target_params, 
                                                self.param_dims)
        eps = 1e-6

        in_a = in_a.double()
        in_b = in_b.double()
        tg_a = tg_a.double()
        tg_b = tg_b.double()

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

        # where its not on the support its inf
        pos_inf = torch.full_like(kl, float('inf'))
        return torch.where(incl, kl, pos_inf)
    
class NormalDistribution(Distribution):
    def __init__(self, ref_parameters):
        super().__init__(ref_parameters)
        self.parametric_kl = True
        self.n_params = 2 # mu, sigma
    
    def constraints(self, params):
        constrained = params.clone()
        # softplus for strict positivity + epsilon 
        # (to avoid killing the log-likelihood with 0 variance)
        constrained[:, 1] = F.softplus(params[:, 1]) + 1e-6
        return constrained
    
    def sample(self, latent_params, batch_size):
        mu, std = split_or_validate_features(latent_params, 
                                          self.param_dims)
        if check_nonbatch(mu):
            # if its only one distirbution we extend to batch_size
            shape = [batch_size, self.sample_dim]
            shape = tuple(shape)
        else:
            shape = [latent_params.size(0), self.sample_dim]
            shape = tuple(shape)
            if batch_size != shape[0]:
                print(f"WARNING: Mismatch between param batch dim ({shape[0]}) and batch_size ({batch_size})!")
                print(f"Ignoring batch_size to prefer the first dimension of params dim: {shape[0]}")
        # Reparametrization trick
        noise = torch.randn(shape, device=mu.device, dtype=mu.dtype)
        return mu + std * noise
    
    def log_likelihood(self, x, params):
        mu, std = split_or_validate_features(params, 
                                             self.param_dims)
        eps = 1e-6

        mu = mu.double(); std = std.double(); x = x.double()
        var = (std**2).clamp_min(eps)

        # feature dimension
        d = mu.size(1) if mu.dim() > 1 else 1

        # log-likelihood for diagonal Gaussian
        log_norm = -0.5 * (d * torch.log(torch.tensor(2.0 * torch.pi)) + torch.log(var).sum(dim=1 if var.dim() > 1 else 0))
        quad = -0.5 * (((x - mu)**2) / var).sum(dim=1 
                                                if var.dim() > 1 
                                                else 0)
        return log_norm + quad
    
    def kl_divergence(self, input_params, target_params):
        mu0, std0 = split_or_validate_features(input_params, self.param_dims)   # p
        mu1, std1 = split_or_validate_features(target_params, self.param_dims)  # q
        eps = 1e-6

        mu0 = mu0.double(); std0 = std0.double()
        mu1 = mu1.double(); std1 = std1.double()

        var0 = (std0**2).clamp_min(eps)
        var1 = (std1**2).clamp_min(eps)

        # feature dimension
        k = mu0.size(1) if mu0.dim() > 1 else 1

        term1 = (var0 / var1).sum(dim=1 if var0.dim() > 1 else 0)
        term2 = (((mu1 - mu0)**2) / var1).sum(dim=1 if var1.dim() > 1 else 0)
        term3 = -k + (torch.log(var1) - torch.log(var0)).sum(dim=1 if var0.dim() > 1 else 0)

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
        self.n_params = 1 # lambda

    def constraints(self, params):
        params = params.double()
        constrained = params.clone()
        # ensure > 0 on lambda
        constrained[:,0] = F.softplus(params[:,0]) + 1e-6
        return constrained
    
    def sample(self, latent_params, batch_size):
        # Same remark as for the NB 
        lmbda = split_or_validate_features(latent_params, 
                                          self.param_dims)
        P = torch.distributions.Poisson(rate=lmbda)
        return P.sample(batch_size)
    
    def log_likelihood(self, x, params):
        # Poisson has a differentiable log likelihood
        lmbda = split_or_validate_features(params, 
                                          self.param_dims)
        P = torch.distributions.Poisson(rate=lmbda)
        return P.log_prob(x)
    
    def kl_divergence(self, input_params, target_params):
        in_l = split_or_validate_features(input_params, 
                                          self.param_dims)
        target_l = split_or_validate_features(target_params, 
                                          self.param_dims)

        # normalize shapes
        in_l = in_l.contiguous().view(-1).double()
        target_l = target_l.contiguous().view(-1).double()

        return in_l * torch.log(1e-8 + in_l / (target_l + 1e-8)) + target_l - in_l

class vonMisesFisher(Distribution):
    def __init__(self, ref_parameters):
        super().__init__(ref_parameters)
        self.parametric_kl = False
        self.n_params = 2 # mu, kappa

    # TODO

class spCauchy(Distribution):
    def __init__(self, ref_parameters):
        super().__init__(ref_parameters)
        self.parametric_kl = False
        self.n_params = 2 # mu, rho

    # TODO