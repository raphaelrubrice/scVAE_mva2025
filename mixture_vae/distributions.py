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

    def get_reference_params(self, z=None):
        """
        Returns concatenated parameters of shape (B, sum(param_dims)).

        Supports two cases for each parameter tensor:
          - (1, D) : standard single prior -> broadcast to (B, D)
          - (K, D) : component-wise prior -> expanded to match a (B*K, ...) input,
                    assuming the caller stacked per-component blocks (k-major order):
                    [k=0 block of B rows, k=1 block of B rows, ...]
        """
        if z is None:
            # return a single concatenated row (1, sum_dims) or (K, sum_dims) if mixture
            # Detect mixture mode by checking if any param has first-dim > 1
            vals = list(self.ref_parameters.values())
            if isinstance(vals[0], torch.Tensor):
                # torch path
                # if (K,D) per param, concat along dim=1 to (K, sum_dims)
                if all(v.ndim == 2 for v in vals) and len({v.size(0) for v in vals}) == 1:
                    return torch.cat([v.contiguous() for v in vals], dim=1)
                # otherwise flatten into (1,sum_dims)
                return torch.cat([v.contiguous().view(1, -1) for v in vals], dim=1)
            else:
                # numpy path
                vals = [np.asarray(v) for v in vals]
                if all(v.ndim == 2 for v in vals) and len({v.shape[0] for v in vals}) == 1:
                    return np.concatenate(vals, axis=1)
                return np.concatenate([v.reshape(1, -1) for v in vals], axis=1)

        assert isinstance(z, (torch.Tensor, np.ndarray)), (
            f"z must be a torch.Tensor or an np.ndarray but got {type(z)}"
        )

        # -----------------------------
        # Torch path
        # -----------------------------
        if isinstance(z, torch.Tensor):
            device, dtype = z.device, z.dtype
            batch_size = z.size(0)

            vals = [v.to(device=device, dtype=dtype).contiguous()
                    for v in self.ref_parameters.values()]

            # Mixture-aware path: each param is (K, D_i) with same K
            if all(v.ndim == 2 for v in vals):
                Ks = [v.size(0) for v in vals]
                if len(set(Ks)) == 1 and Ks[0] > 1:
                    K = Ks[0]
                    base = torch.cat(vals, dim=1)  # (K, sum_dims)

                    # We expect batch_size == B*K with k-major stacking
                    if batch_size % K == 0:
                        B = batch_size // K
                        # repeat each component row B times in order: 0..K-1
                        return base.repeat_interleave(B, dim=0)  # (B*K, sum_dims)

                    # Fallback: cannot infer mapping; broadcast first component
                    return base[:1].expand(batch_size, -1)

            # Standard path: flatten each into one row then expand to (B, sum_dims)
            base = torch.cat([v.view(1, -1) for v in vals], dim=1)  # (1, sum_dims)
            return base.expand(batch_size, -1)

        # -----------------------------
        # NumPy path
        # -----------------------------
        batch_size = z.shape[0]
        vals = [np.asarray(v) for v in self.ref_parameters.values()]

        if all(v.ndim == 2 for v in vals):
            Ks = [v.shape[0] for v in vals]
            if len(set(Ks)) == 1 and Ks[0] > 1:
                K = Ks[0]
                base = np.concatenate(vals, axis=1)  # (K, sum_dims)

                if batch_size % K == 0:
                    B = batch_size // K
                    return np.repeat(base, repeats=B, axis=0)  # (B*K, sum_dims)

                return np.broadcast_to(base[:1], (batch_size, base.shape[1]))

        base = np.concatenate([v.reshape(1, -1) for v in vals], axis=1)
        return np.broadcast_to(base, (batch_size, base.shape[1]))
    

class CategoricalDistribution(Distribution):
    def __init__(self, ref_parameters):
        super().__init__(ref_parameters)
        self.parametric_kl = True
        self.n_params = 1
        self.eps = 1e-12

    def _normalize(self, p: torch.Tensor) -> torch.Tensor:
        p = p.clamp_min(self.eps)
        return p / p.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def get_ref_proba(self):
        probs = self.ref_parameters["probs"]
        if isinstance(probs, torch.Tensor) and probs.ndim == 2 and probs.size(0) == 1:
            probs = probs.squeeze(0)
        return probs

    def get_reference_params(self, z=None):
        """
        For categorical, params are naturally dict-shaped.
        (This does not affect the continuous distributions, which remain concatenated.)
        """
        probs = self.get_ref_proba()
        if z is None:
            return {"probs": probs}

        if isinstance(z, torch.Tensor):
            probs = probs.to(device=z.device, dtype=z.dtype)
            if probs.ndim == 1:
                probs = probs.unsqueeze(0)
            return {"probs": probs.expand(z.size(0), -1)}

        probs = np.asarray(probs)
        if probs.ndim == 2 and probs.shape[0] == 1:
            probs = probs.squeeze(0)
        return {"probs": np.broadcast_to(probs.reshape(1, -1), (z.shape[0], probs.shape[0]))}

    def sample(self, latent_params=None, batch_size=1):
        if isinstance(latent_params, dict):
            probs = latent_params["probs"]
        else:
            probs = self.get_ref_proba()

        if not isinstance(probs, torch.Tensor):
            probs = torch.as_tensor(probs)
        probs = self._normalize(probs)

        if probs.ndim == 1:
            return torch.multinomial(probs, num_samples=batch_size, replacement=True)

        if probs.size(0) != batch_size:
            batch_size = probs.size(0)
        return torch.multinomial(probs, num_samples=1, replacement=True).squeeze(-1)

    def log_likelihood(self, x, params):
        probs = self._normalize(params["probs"])
        if x.ndim == 2 and x.size(-1) == 1:
            x = x.squeeze(-1)
        x = x.long()

        if probs.ndim == 1:
            probs = probs.unsqueeze(0)
        logp = probs.clamp_min(self.eps).log()
        return logp.gather(dim=-1, index=x.unsqueeze(-1)).squeeze(-1)

    def kl_divergence(self, input_params, target_params):
        q = input_params["probs"]
        p = target_params["probs"]
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if p.ndim == 1:
            p = p.unsqueeze(0)

        q = self._normalize(q)
        p = self._normalize(p)
        return (q * (q.clamp_min(self.eps).log() - p.clamp_min(self.eps).log())).sum(dim=-1)



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

    def constraints(self, params):
        params = params.double()
        constrained = params.clone()
        # ensure positivity on r
        r_dims = self.param_dims[1]
        logits_dims = self.param_dims[0]
        start, end = logits_dims, r_dims + logits_dims
        constrained[:,start:end] = F.softplus(params[:,start:end]) + 1e-6
        return constrained

    def sample(self, latent_params, batch_size):
        # Sampling is complex for NB, so we keep the object here for safety/correctness
        # as it is rarely used in the inner training loop (mostly for evaluation).
        logits, r = split_or_validate_features(latent_params, self.param_dims)
        NB = torch.distributions.NegativeBinomial(total_count=r, logits=logits)
        return NB.sample((batch_size,))

    def log_likelihood(self, x, params):
        """
        Optimized functional log_prob implementation.

        PyTorch-style parameterization:
        p = sigmoid(logits)
        P(k) = Gamma(k + r) / (Gamma(k + 1) Gamma(r)) * (1 - p)^r * p^k
        """
        logits, r = split_or_validate_features(params, self.param_dims)
        
        # log_prob = lgamma(x + r) - lgamma(r) - lgamma(x + 1)
        #            + r * log(1 - p) + x * log(p)
        # log(p)     = log_sigmoid(logits)
        # log(1 - p) = log_sigmoid(-logits)

        log_unnormalized_prob = (
            r * F.logsigmoid(-logits) +  # r * log(1 - p)
            x * F.logsigmoid(logits)     # x * log(p)
        )

        log_normalization = (
            torch.lgamma(x + r) -
            torch.lgamma(r) -
            torch.lgamma(x + 1)
        )

        log_p = log_unnormalized_prob + log_normalization
        return log_p

    def kl_divergence(self, input_params, target_params):
        pass

class Poisson(Distribution):
    def __init__(self, ref_parameters):
        super().__init__(ref_parameters)
        self.parametric_kl = False
        self.n_params = 1  # lambda

    def constraints(self, params):
        params = params.double()
        constrained = params.clone()
        constrained[:, 0] = F.softplus(params[:, 0]) + 1e-6
        return constrained

    def sample(self, latent_params, batch_size):
        lmbda = split_or_validate_features(latent_params, self.param_dims)
        P = torch.distributions.Poisson(rate=lmbda)
        return P.sample((batch_size,))

    def log_likelihood(self, x, params):
        """
        Optimized functional log_prob implementation.
        log P(x|lambda) = x*log(lambda) - lambda - lgamma(x+1)
        """
        lmbda = split_or_validate_features(params, self.param_dims)
        
        # Add epsilon to lambda inside log to avoid NaN if lambda approaches 0
        # though constraints usually handle this.
        log_p = x * torch.log(lmbda + 1e-10) - lmbda - torch.lgamma(x + 1)
        
        return log_p

    def kl_divergence(self, input_params, target_params):
        in_l = split_or_validate_features(input_params, self.param_dims)
        target_l = split_or_validate_features(target_params, self.param_dims)
        device = in_l.device
        dtype = in_l.dtype
        in_l = in_l.contiguous().view(-1).to(device=device, dtype=dtype)
        target_l = target_l.contiguous().view(-1).to(device=device, dtype=dtype)
        return in_l * torch.log(1e-8 + in_l / (target_l + 1e-8)) + target_l - in_l

class Student(Distribution):
    def __init__(self, ref_parameters):
        super().__init__(ref_parameters)
        self.parametric_kl = False
        self.n_params = 3 # df, mu, scale

    def constraints(self, params):
        params = params.double()
        constrained = params.clone()
        
        dim_df = self.param_dims[0]
        dim_mu = self.param_dims[1]
        
        # df > 0
        constrained[:, :dim_df] = F.softplus(params[:, :dim_df]) + 1e-6
        # scale > 0
        start_scale = dim_df + dim_mu
        constrained[:, start_scale:] = F.softplus(params[:, start_scale:]) + 1e-6
        
        return constrained

    def sample(self, latent_params, batch_size):
        """
        Optimized reparameterized sampling:
        X = mu + scale * (Z / sqrt(V/df))
        Z ~ N(0,1), V ~ Chi2(df)
        """
        df, mu, scale = split_or_validate_features(latent_params, self.param_dims)
        
        if check_nonbatch(df):
            shape = (batch_size, self.sample_dim)
        else:
            shape = tuple(list(latent_params.size())[:-1] + [self.sample_dim])
            
        device = df.device
        dtype = df.dtype
        
        # Z ~ Normal(0, 1)
        z = torch.randn(shape, device=device, dtype=dtype)
        
        
        gamma_dist = D.Gamma(concentration=df / 2.0, rate=0.5)
        v = gamma_dist.rsample(sample_shape=shape if df.shape[0]==1 else ())
        # Note: if df has batch dim, rsample handles it.
        
        # If df was broadcasted, v will match.
        
        x = mu + scale * (z * torch.rsqrt(v / df))
        return x

    def log_likelihood(self, x, params):
        """
        Optimized functional log_prob for Student T.
        log p(x) = log Gamma((v+1)/2) - log Gamma(v/2) - 0.5 log(pi*v) - log(scale)
                   - (v+1)/2 * log(1 + (1/v)*((x-mu)/scale)^2)
        """
        df, mu, scale = split_or_validate_features(params, self.param_dims)
        
        # Constants
        pi = torch.tensor(np.pi, device=x.device, dtype=x.dtype)
        
        # Mahalanobis distance squared
        z = (x - mu) / scale
        log_term = torch.log1p( (z**2) / df )
        
        # Log Norm
        # lgamma returns log(Gamma(x))
        log_norm = (torch.lgamma((df + 1) / 2) 
                    - torch.lgamma(df / 2) 
                    - 0.5 * torch.log(df * pi) 
                    - torch.log(scale))
        
        log_p = log_norm - 0.5 * (df + 1) * log_term
        
        if log_p.dim() > 1:
            return log_p.sum(dim=1)
        return log_p

    def kl_divergence(self, input_params, target_params):
        pass