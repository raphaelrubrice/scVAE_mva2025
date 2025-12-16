"""Sub-module to define the model classes of Mixture VAEs"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import product
import time

from mixture_vae.distributions import Distribution


class BaseBlock(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 n_layers: int = 2,
                 act_func: callable = nn.ReLU(),
                 final_act_func: callable = None,
                 dropout: float = 0.0,
                 norm_layer: callable = nn.LayerNorm):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.act_func = act_func
        self.final_act_func = self.act_func if final_act_func is None else final_act_func
        self.dropout = dropout
        self.norm_layer = norm_layer

        self.dense_block = nn.ModuleList()

        if self.norm_layer is None:
            # +1 because n_layers is the number of hidden
            for i in range(self.n_layers+1):
                if i == 0:
                    if self.dropout > 0:
                        module = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                        self.act_func,
                                        nn.Dropout(self.dropout))
                    else:
                        module = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                        self.act_func,
                                        )
                elif i == self.n_layers:
                    module = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim),
                                            self.final_act_func)
                else:
                    if self.dropout > 0:
                        module = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                        self.act_func,
                                        nn.Dropout(self.dropout))
                    else:
                        module = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                        self.act_func,
                                        )
                self.dense_block.append(module)
        else:
            # +1 because n_layers is the number of hidden
            for i in range(self.n_layers+1):
                if i == 0:
                    if self.dropout > 0:
                        module = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                        self.act_func,
                                        self.norm_layer(self.hidden_dim),
                                        nn.Dropout(self.dropout))
                    else:
                        module = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                        self.act_func,
                                        self.norm_layer(self.hidden_dim))
                elif i == self.n_layers:
                    module = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim),
                                            self.final_act_func)
                else:
                    if self.dropout > 0:
                        module = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                        self.act_func,
                                        self.norm_layer(self.hidden_dim),
                                        nn.Dropout(self.dropout))
                    else:
                        module = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                        self.act_func,
                                        self.norm_layer(self.hidden_dim))
                self.dense_block.append(module)
            
    def forward(self, x):
        # B x input_dim
        for layer in self.dense_block:
            x = layer(x) 
        return x # B x output_dim
                
def broadcast_cat_probs(ref, like, eps=1e-12, allow_batched=True):
    ref = ref.to(device=like.device, dtype=like.dtype)

    B, K = like.size(0), like.size(1)

    # Normalize shapes to (1, K) or (B, K)
    if ref.ndim == 1:
        # (K,)
        if ref.numel() != K:
            raise ValueError(f"Expected K={K} probs, got {ref.numel()} in shape {tuple(ref.shape)}")
        ref = ref.unsqueeze(0)  # (1,K)

    elif ref.ndim == 2:
        if ref.shape == (1, K):
            pass  # already (1,K)
        elif ref.shape == (K, 1):
            ref = ref.view(1, K)  # (K,1) -> (1,K)
        elif allow_batched and ref.shape == (B, K):
            # already batched
            pass
        else:
            raise ValueError(
                f"Expected ref probs shape (K,), (1,K), (K,1)"
                f"{' or (B,K)' if allow_batched else ''}, got {tuple(ref.shape)}; expected K={K}, B={B}"
            )
    else:
        raise ValueError(f"Expected ref probs ndim 1 or 2, got ndim={ref.ndim} with shape {tuple(ref.shape)}")

    # Clamp + renormalize along K
    ref = ref.clamp_min(eps)
    ref = ref / ref.sum(dim=-1, keepdim=True).clamp_min(eps)

    # Expand to (B,K) if needed
    if ref.shape == (1, K):
        ref = ref.expand(B, K)

    return ref

class MixtureVAE(nn.Module):
    """
    Reimplementing the architecture presented in scVAE
    Where the latent variables are assumed to be drawn from
    a Mixture of some Distribution. 
    This class is flexible because it allows to define priors
    and posteriors as desired.
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 n_components: int,
                 n_layers: int,
                 prior_latent: Distribution, # Normal
                 prior_input: Distribution, # Negative Binomial
                 prior_categorical: Distribution, # uniform
                 posterior_latent: Distribution, # Normal
                 act_func: callable = nn.ReLU(),
                 dropout: int = 0.0,
                 norm_layer: callable = nn.LayerNorm,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_components = n_components
        self.n_levels = 1
        self.n_layers = n_layers
        self.act_func = act_func
        self.dropout = dropout
        self.norm_layer = norm_layer
        self.prior_latent = prior_latent
        self.prior_input = prior_input
        assert hasattr(prior_categorical, "get_ref_proba"), f"Missing method: The prior for built-in clustering must be a categorical distribution (i.e have the get_ref_proba method)"
        self.prior_categorical = prior_categorical
        self.posterior_latent = posterior_latent

        # Module to approximate p(y) = Clustering module
        self.clustering_block = BaseBlock(input_dim=self.input_dim,
                                           hidden_dim=self.hidden_dim,
                                           output_dim=self.n_components,
                                           n_layers=self.n_layers,
                                           act_func=self.act_func,
                                           final_act_func=nn.Softmax(dim=1), # probabilities
                                           dropout=self.dropout,
                                           norm_layer=self.norm_layer)
        
        # Modules to approximate q(z|x,y) = Latent modules
        # parameter constraints (if any) are enforced during forward pass
        # One module for each distribution parameter
        self.encoder = nn.ModuleList()
        for param_dim in self.posterior_latent.param_dims:
            param_encoder = BaseBlock(input_dim=self.input_dim + self.n_components,
                                    hidden_dim=self.hidden_dim,
                                    output_dim=param_dim,
                                    n_layers=self.n_layers,
                                    act_func=self.act_func,
                                    dropout=self.dropout,
                                    norm_layer=self.norm_layer)
            self.encoder.append(param_encoder)
        self.latent_dim = self.posterior_latent.sample_dim

        # Modules to learn p(x|z) = Generation modules
        # ATTENTION: in scVAE we dont reconstruct the input directly, we 
        # output the parameters of the distribution it is supposed to follow
        # parameter constraints (if any) are enforced during forward pass
        # One module for each distribution parameter
        self.decoder = nn.ModuleList()
        for param_dim in self.prior_input.param_dims:
            param_decoder = BaseBlock(input_dim=self.latent_dim,
                                    hidden_dim=self.hidden_dim,
                                    output_dim=param_dim,
                                    n_layers=self.n_layers,
                                    act_func=self.act_func,
                                    dropout=self.dropout,
                                    norm_layer=self.norm_layer)
            self.decoder.append(param_decoder)
    
    def compute_encoding_params(self, x):
        tensor_list = []
        for param_encoder in self.encoder:
            tensor_list.append(param_encoder(x))
        return torch.cat(tensor_list, dim=1)
    
    def compute_decoding_params(self, x):
        tensor_list = []
        for param_decoder in self.decoder:
            tensor_list.append(param_decoder(x))
        return torch.cat(tensor_list, dim=1)

    
    def encode(self, x, at_level=-1): 
        # at_level exists to simplify compatibility with other models
        BATCH_SIZE = x.size(0)
        # B x n_components
        cluster_probas = self.clustering_block(x)

        # build all n_components one-hots once, 
        # then expand to batch
        eyeK = torch.eye(self.n_components, device=x.device)
        all_z = []
        all_latent = []
        all_ck = [eyeK[k].expand(x.size(0), -1) for k in range(self.n_components)]
        all_enc_in = [torch.cat([x, c_k], dim=1) for c_k in all_ck]
        all_enc_in = torch.cat(all_enc_in)
        # For each cluster we need to compute the z (mixture)
        # get latent dist parameters
        all_latent_k = self.posterior_latent.constraints(self.compute_encoding_params(all_enc_in))
        for i in range(self.n_components):
            all_latent.append(all_latent_k[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]) # params for q_k(z|x,c=k)

        # sample 1 time for each sample in the batch
        # = get the latent sample under the condition y = k
        all_z_k = self.posterior_latent.sample(all_latent_k, x.size(0)*self.n_components)
        for i in range(self.n_components):
            all_z.append(all_z_k[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]) # sample from q_k

        z_mixture = torch.cat([t.unsqueeze(1) for t in all_z], dim=1) # B x K x latent_dim
        z = (cluster_probas.unsqueeze(-1) * z_mixture).sum(dim=1) # B x latent_dim

        # latent_params: same "mixture-averaged" params (rarely needed downstream)
        lp = torch.stack(all_latent, dim=1)                    # B x K x (Dz * n_params/posterior)
        latent_params = (cluster_probas.unsqueeze(-1) * lp).sum(dim=1)

        return z, latent_params, cluster_probas, all_z, all_latent
    
    def decode(self, z):
        input_params = self.compute_decoding_params(z)
        # enforce constraints on params (required for some distributions)
        input_params = self.prior_input.constraints(input_params)
        return input_params
    
    def forward(self, x):
        z, latent_params, cluster_probas, all_z, all_latent = self.encode(x)

        input_params = self.decode(z)
        return input_params, z, latent_params, cluster_probas, all_z, all_latent
    
    def cluster_input(self, x=None, cluster_probas=None):
        if x is None:
            assert cluster_probas is not None, f"Missing cluster_probas: When given no input features you must provide precomputed cluster_probas"
        else:
            cluster_probas = self.encode(x)[2]
        return torch.argmax(cluster_probas, dim=1)

    def log_likelihood_input(self, x, params):
        return self.prior_input.log_likelihood(x, params)
    
    def kl_div(self, z=None, learned_params=None):
        if z is None:
            assert learned_params is not None, "learned_params required when z is None"
            prior_params = self.prior_latent.get_reference_params(learned_params)
            if self.posterior_latent.parametric_kl:
                # KL(q_post || p_prior)
                kl = self.posterior_latent.kl_divergence(learned_params, prior_params)
                # sum over latent dims, keep batch dim
                return kl
            else:
                # Monte Carlo estimation: sample z ~ q 
                # tehn comput log q - log p (average outside)
                B = learned_params.size(0)
                z = self.posterior_latent.sample(learned_params, B)
                log_q = self.posterior_latent.log_likelihood(z, learned_params)
                log_p = self.prior_latent.log_likelihood(z, prior_params)
                return (log_q - log_p)
        else:
            # Monte Carlo estimate per z
            prior_params = self.prior_latent.get_reference_params(z)
            log_q = self.posterior_latent.log_likelihood(z, learned_params)
            log_p = self.prior_latent.log_likelihood(z, prior_params)
            # (average outside)
            return (log_q - log_p) # because kl loss = Eq[log q_z/p_z] and the average is done outside

    def iwae(self, batch_x, N = 500):
        """
        Estimates total marginal log-likelihood using the 
        IWAE from Burga et al. in 2016
        """
        self.eval()
        print(batch_x.device)
        with torch.no_grad():
            (z, 
            latent_params, 
            cluster_probas, 
            all_z, 
            all_latent
            ) = self.encode(batch_x)
            
            batch_size = batch_x.size(0)
            clusters = torch.argmax(cluster_probas,dim=1)

            # fetch latent from the appropriate cluster for each input sample
            selected_latents = torch.cat([all_latent[c][i:i+1,:] 
                                        for i,c in enumerate(clusters)], 
                                        dim=0)

            sum_paramdims = selected_latents.size(1)
            latent_params = selected_latents.unsqueeze(1) # batch, 1, sum paramdims
            latent_params = latent_params.expand(batch_size, N, sum_paramdims)

            # sample from latent posterior
            posterior_latent = self.posterior_latent
            sampled_z = posterior_latent.sample(latent_params, batch_size)

            # obtain decoder output (parameters of the prior distirbution for each sample)
            latent_dim = self.latent_dim

            z_flat = sampled_z.reshape(batch_size * N, latent_dim)
            x_params = self.decode(z_flat) 

            # Align x with decoder params
            # If likelihood is per-feature (e.g. NB for genes), x is (B, G).
            # Repeat each x_i N times to match (B*N, G).
            x_rep = batch_x.unsqueeze(1).expand(batch_size, N, batch_x.size(1)).reshape(batch_size * N, -1)

            # log p(x|z,y): should depend on z through x_params
            log_p_x_zy = self.prior_input.log_likelihood(x_rep, x_params)
            # If log_likelihood returns per-feature contributions, sum over features:
            if log_p_x_zy.ndim > 1:
                log_p_x_zy = log_p_x_zy.sum(dim=1)
            log_p_x_zy = log_p_x_zy.reshape(batch_size, N)

            # log p(z)
            prior_latent = self.prior_latent
            prior_latent_params = prior_latent.get_reference_params(sampled_z.view(-1,latent_dim))
            log_p_z = prior_latent.log_likelihood(sampled_z.view(-1,latent_dim), 
                                                       prior_latent_params)
            log_p_z = log_p_z.reshape(batch_size, N)
            
            # log q(z|x,y)
            log_q_z_xy = posterior_latent.log_likelihood(sampled_z.view(-1, latent_dim),
                                                        latent_params.reshape(-1, latent_params.size(-1))
                                                        )
            log_q_z_xy = log_q_z_xy.reshape(batch_size, N)
            log_q_z_xy = log_q_z_xy.reshape(batch_size, N)

            # Log sum exp trick for stable compute of the likelihood ratios
            logspace_ratio = log_p_x_zy + log_p_z - log_q_z_xy
            sumexp = torch.logsumexp(logspace_ratio, dim=1)
            return sumexp - torch.log(torch.tensor([N], device=sumexp.device)) # IWAE for each sample in batch_x

def kl_marginal_usage(level_probas: torch.Tensor,
                      ref_probas: torch.Tensor,
                      eps: float = 1e-12) -> torch.Tensor:
    """
    used to encourage usage of all components
    level_probas: (B,K) probabilities q(y|x)
    ref_probas:   (K,) or (1,K) reference categorical probs p(y)
    returns: scalar KL( mean_x q(y|x) || ref )
    """
    qbar = level_probas.mean(dim=0)                       # (K,)
    qbar = qbar.clamp_min(eps)

    if ref_probas.ndim == 2:
        ref = ref_probas.squeeze(0)
    else:
        ref = ref_probas
    ref = ref.to(qbar.device).clamp_min(eps)              # (K,)

    kl = (qbar * (qbar.log() - ref.log())).sum()
    return kl

# def elbo_mixture_step(model: MixtureVAE,
#                       x: torch.Tensor,
#                       beta_kl: float = 1.0,
#                       reg_marginal: float = 1.0,
#                       track_clusters: bool = False,
#                       n_levels: int = 1):
#     # Forward (keeps all component samples)
#     (input_params,
#      z_mixture,
#      latent_params_mixture,
#      cluster_probas,
#      all_z,
#      all_latent) = model(x)

#     # clusters
#     if track_clusters:
#         clusters = model.cluster_input(cluster_probas=cluster_probas)
#     else:
#         clusters = None

#     BATCH_SIZE = x.size(0)
#     K = model.n_components

#     # 1) Reconstruction
#     all_zk = torch.cat([all_z[k] for k in range(K)], dim=0)
#     all_params_k = model.decode(all_zk)
#     log_px_per_k = model.log_likelihood_input(x.repeat(K, 1), all_params_k)
#     log_px_per_k = torch.stack(
#         [log_px_per_k[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :] for i in range(K)],
#         dim=1
#     )  # (B, K, input_dim)
#     recon = (cluster_probas.unsqueeze(2) * log_px_per_k).sum(dim=1).sum(dim=1).mean()

#     # 2) Latent KL
#     all_latent_k = torch.cat([all_latent[k] for k in range(K)], dim=0)
#     kl_per_k = model.kl_div(z=None, learned_params=all_latent_k)
#     if kl_per_k.dim() > 1:
#         kl_per_k = kl_per_k.sum(dim=1)
#     kl_per_k = torch.stack(
#         [kl_per_k[i*BATCH_SIZE:(i+1)*BATCH_SIZE] for i in range(K)],
#         dim=1
#     )  # (B, K)
#     kl_z = (cluster_probas * kl_per_k).sum(dim=1).mean()

#     # 3) Cluster KL (per-sample)
#     ref = broadcast_cat_probs(model.prior_categorical.get_ref_proba(), cluster_probas)
#     kl_pi = (cluster_probas * (cluster_probas.clamp_min(1e-12).log()
#                                - ref.clamp_min(1e-12).log())).sum(dim=1).mean()

#     # 4) Marginal usage KL (dataset-level within batch)
#     kl_marg = torch.zeros((), device=cluster_probas.device, dtype=cluster_probas.dtype)
#     if reg_marginal is not None and reg_marginal > 0:
#         # Use the non-broadcast reference probs for KL(qbar || ref)
#         ref_probs = model.prior_categorical.get_ref_proba()
#         if not isinstance(ref_probs, torch.Tensor):
#             ref_probs = torch.tensor(ref_probs, device=cluster_probas.device, dtype=cluster_probas.dtype)
#         kl_marg = kl_marginal_usage(cluster_probas, ref_probs)

#     # normalize per level (keep kl_marg un-normalized by default; you can choose otherwise)
#     recon = recon / n_levels
#     kl_z = kl_z / n_levels
#     kl_pi = kl_pi / n_levels

#     elbo = recon - beta_kl * (kl_z + kl_pi) - reg_marginal * kl_marg
#     loss = -elbo

#     return loss, dict(
#         recon=recon.detach(),
#         kl_latent=kl_z.detach(),
#         kl_cluster=kl_pi.detach(),
#         kl_marginal=kl_marg.detach(),
#     ), clusters


# def summed_elbo_mixture_step(model,
#                             x,
#                             beta_kl: float | None = None,
#                             reg_marginal: float = 1.0,
#                             track_clusters: bool = False):
#     if beta_kl is None:
#         betas_kl = [1.0 for _ in range(len(model.branches))]
#     else:
#         betas_kl = [float(beta_kl) for _ in range(len(model.branches))]

#     P = {
#         "recon": 0.0,
#         "kl_latent": 0.0,
#         "kl_cluster": 0.0,
#         "kl_marginal": 0.0,
#     }

#     L = 0.0
#     clusters = None

#     for idx, (m, beta) in enumerate(zip(model.branches, betas_kl)):
#         track_clusters_level = bool(track_clusters and idx == model.n_levels - 1)

#         loss_value, parts, batch_clusters = elbo_mixture_step(
#             m, x,
#             beta_kl=beta,
#             reg_marginal=reg_marginal,
#             track_clusters=track_clusters_level,
#             n_levels=model.n_levels
#         )

#         L += (loss_value / len(model.branches))

#         if batch_clusters is not None:
#             clusters = batch_clusters

#         for pname in parts:
#             P[pname] += (parts[pname] / len(model.branches))

#     return L, P, clusters

def elbo_mixture_step(model: MixtureVAE,
                      x: torch.Tensor,
                      beta_kl: float = 1.0,
                      reg_marginal: float = 1.0,
                      track_clusters: bool = False,
                      n_levels: int = 1,
                      scale_marg=None):
    # Forward (keeps all component samples)
    (input_params,
     z_mixture,
     latent_params_mixture,
     cluster_probas,
     all_z,
     all_latent) = model(x)

    # clusters
    if track_clusters:
        clusters = model.cluster_input(cluster_probas=cluster_probas)
    else:
        clusters = None

    BATCH_SIZE = x.size(0)
    K = model.n_components

    # 1) Reconstruction
    all_zk = torch.cat([all_z[k] for k in range(K)], dim=0)
    all_params_k = model.decode(all_zk)
    log_px_per_k = model.log_likelihood_input(x.repeat(K, 1), all_params_k)
    log_px_per_k = torch.stack(
        [log_px_per_k[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :] for i in range(K)],
        dim=1
    )  # (B, K, input_dim)
    recon = (cluster_probas.unsqueeze(2) * log_px_per_k).sum(dim=1).sum(dim=1).mean()

    # 2) Latent KL
    all_latent_k = torch.cat([all_latent[k] for k in range(K)], dim=0)
    kl_per_k = model.kl_div(z=None, learned_params=all_latent_k)
    if kl_per_k.dim() > 1:
        kl_per_k = kl_per_k.sum(dim=1)
    kl_per_k = torch.stack(
        [kl_per_k[i*BATCH_SIZE:(i+1)*BATCH_SIZE] for i in range(K)],
        dim=1
    )  # (B, K)
    kl_z = (cluster_probas * kl_per_k).sum(dim=1).mean()

    # 3) Cluster KL (per-sample)
    ref = broadcast_cat_probs(model.prior_categorical.get_ref_proba(), cluster_probas)
    kl_pi = (cluster_probas * (cluster_probas.clamp_min(1e-12).log()
                               - ref.clamp_min(1e-12).log())).sum(dim=1).mean()

    # 4) Marginal usage KL (dataset-level within batch)
    kl_marg = torch.zeros((), device=cluster_probas.device, dtype=cluster_probas.dtype)
    if reg_marginal is not None and reg_marginal > 0:
        # Use the non-broadcast reference probs for KL(qbar || ref)
        ref_probs = model.prior_categorical.get_ref_proba()
        if not isinstance(ref_probs, torch.Tensor):
            ref_probs = torch.tensor(ref_probs, device=cluster_probas.device, dtype=cluster_probas.dtype)
        kl_marg = kl_marginal_usage(cluster_probas, ref_probs)

    # normalize per level (keep kl_marg un-normalized by default; you can choose otherwise)
    recon = recon / n_levels
    kl_z = kl_z / n_levels
    kl_pi = kl_pi / n_levels
    if scale_marg is not None:
        kl_marg = kl_marg * scale_marg
    elbo = recon - beta_kl * (kl_z + kl_pi) - reg_marginal * kl_marg
    loss = -elbo

    return loss, dict(
        recon=recon.detach(),
        kl_latent=kl_z.detach(),
        kl_cluster=kl_pi.detach(),
        kl_marginal=kl_marg.detach(),
    ), clusters


def summed_elbo_mixture_step(model,
                            x,
                            beta_kl: float | None = None,
                            reg_marginal: float = 1.0,
                            track_clusters: bool = False):
    if beta_kl is None:
        betas_kl = [1.0 for _ in range(len(model.branches))]
    else:
        betas_kl = [float(beta_kl) for _ in range(len(model.branches))]

    P = {
        "recon": 0.0,
        "kl_latent": 0.0,
        "kl_cluster": 0.0,
        "kl_marginal": 0.0,
    }

    L = 0.0
    clusters = None

    for idx, (m, beta) in enumerate(zip(model.branches, betas_kl)):
        track_clusters_level = bool(track_clusters and idx == model.n_levels - 1)

        loss_value, parts, batch_clusters = elbo_mixture_step(
            m, x,
            beta_kl=beta,
            reg_marginal=reg_marginal,
            track_clusters=track_clusters_level,
            n_levels=1,
            scale_marg=len(model.branches),
        )

        L += (loss_value / len(model.branches))

        if batch_clusters is not None:
            clusters = batch_clusters

        for pname in parts:
            P[pname] += (parts[pname] / len(model.branches))

    return L, P, clusters

class ind_MoMVAE(nn.Module):
    """
    
    """
    
    def __init__(self, PARAMS: list[dict[str, float]]) -> None:
        """
        
        """
        super(ind_MoMVAE, self).__init__()
        self.branches = nn.ModuleList([MixtureVAE(**params) for params in PARAMS])
        self.n_levels = len(self.branches)
        self.last_level = self.n_levels - 1
        return None
    
    def encode(self, x, at_level=None):
        if at_level is None:
            at_level = self.n_levels - 1
        else:
            assert at_level >= 0 and isinstance(at_level, int), f"at_level must be an index, i.e, a positive int but got {at_level}"
        self.last_level = at_level
        return self.branches[at_level].encode(x)

    def decode(self, z):
        return self.branches[self.last_level].decode(z)
    
    def forward(self, x, at_level=None):
        (z, 
         latent_params, 
         cluster_probas, 
         all_z, 
         all_latent
         ) = self.encode(x, at_level=at_level)

        input_params = self.decode(z)
        return input_params, z, latent_params, cluster_probas, all_z, all_latent
    
    def cluster_input(self, x=None, cluster_probas=None, at_level=None): # AFFECTED
        if x is None:
            assert cluster_probas is not None, f"Missing cluster_probas: When given no input features you must provide precomputed cluster_probas"
        else:
            cluster_probas = self.encode(x, at_level=at_level)[2]
        return torch.argmax(cluster_probas, dim=1)
    
    def iwae(self, batch_x, N = 500, at_level=None):
        """
        Estimates total marginal log-likelihood using the 
        IWAE from Burga et al. in 2016
        """
        if at_level is None:
            at_level = self.n_levels - 1
        return self.branches[at_level].iwae(batch_x)


class MoMixVAE(nn.Module):
    """
    Implementing a Mixture of Mixtures VAE (MoMixVAE) as a framework
    for built-in hierarchical clustering.
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 hierarchy_components: list,
                 n_layers: int,
                 all_prior_latent: list[Distribution], # Normal
                 prior_input: Distribution, # Negative Binomial
                 all_prior_categorical: list[Distribution], # uniform
                 all_posterior_latent: list[Distribution], # Normal
                 act_func: callable = nn.ReLU(),
                 dropout: int = 0.0,
                 norm_layer: callable = nn.LayerNorm, # batchnorm prevents parallelization 
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # check for valid hierarchy
        self.check_hierarchy(hierarchy_components)
        self.hierarchy_components = hierarchy_components
        self.n_levels = len(self.hierarchy_components)
        # create combinations at each level
        self.combinations = self.make_combinations()
        self.n_layers = n_layers
        self.act_func = act_func
        self.dropout = dropout
        self.norm_layer = norm_layer

        assert len(all_prior_latent) == len(hierarchy_components), "all_prior_latent must have one prior per hierarchy level"
        self.all_prior_latent = all_prior_latent

        self.prior_input = prior_input

        assert sum([hasattr(prior, "get_ref_proba") for prior in all_prior_categorical]) == len(all_prior_categorical), f"Missing method: All priors for built-in clustering must be a categorical distribution (i.e have the get_ref_proba method), at least one did not"
        self.all_prior_categorical = all_prior_categorical # list of priors !
        
        assert len(np.unique([post.sample_dim for post in all_posterior_latent])) == 1, f"Currently, posteriors on latent Z shoudl all share the same dimension, at least one differ"
        self.all_posterior_latent = all_posterior_latent # list of posteriors !

        # Module to approximate p(y) = Clustering module
        self.clustering_block = nn.ModuleList()
        for level in range(self.n_levels):
            if level == 0:
                cluster_dims = 0
            else:
                cluster_dims = sum(self.hierarchy_components[:level])
            # print("Clustering: Level", level, "cluster_dims", cluster_dims)
            cluster_module = BaseBlock(input_dim=self.input_dim + cluster_dims,
                                        hidden_dim=self.hidden_dim,
                                        output_dim=self.hierarchy_components[level],
                                        n_layers=self.n_layers,
                                        act_func=self.act_func,
                                        final_act_func=nn.Softmax(dim=1), # probabilities
                                        dropout=self.dropout,
                                        norm_layer=self.norm_layer)
            self.clustering_block.append(cluster_module)
        
        # Modules to approximate q(z|x,y) = Latent modules
        # parameter constraints (if any) are enforced during forward pass
        # One module for each distribution parameter
        # for each level of clustering
        self.encoder = nn.ModuleList()
        for level in range(self.n_levels):
            enc_level = nn.ModuleList()
            cluster_dims = sum(self.hierarchy_components[:level+1])
            # print("Encoder: Level", level, "cluster_dims", cluster_dims)
            for param_dim in self.all_posterior_latent[level].param_dims:
                param_encoder = BaseBlock(input_dim=self.input_dim + cluster_dims,
                                        hidden_dim=self.hidden_dim,
                                        output_dim=param_dim,
                                        n_layers=self.n_layers,
                                        act_func=self.act_func,
                                        dropout=self.dropout,
                                        norm_layer=self.norm_layer)
                enc_level.append(param_encoder)
            self.encoder.append(enc_level)
        self.latent_dim = self.all_posterior_latent[0].sample_dim # they should all share the same latent_dim 

        # Modules to learn p(x|z) = Generation modules
        # ATTENTION: in scVAE we dont reconstruct the input directly, we 
        # output the parameters of the distribution it is supposed to follow
        # parameter constraints (if any) are enforced during forward pass
        # One module for each distribution parameter
        self.decoder = nn.ModuleList()
        for param_dim in self.prior_input.param_dims:
            param_decoder = BaseBlock(input_dim=self.latent_dim,
                                    hidden_dim=self.hidden_dim,
                                    output_dim=param_dim,
                                    n_layers=self.n_layers,
                                    act_func=self.act_func,
                                    dropout=self.dropout,
                                    norm_layer=self.norm_layer)
            self.decoder.append(param_decoder)
    
    def check_hierarchy(self, hierarchy):
        """
        To ensure a valid hierarchy
        """
        assert hierarchy[0] >= 2, f"Highest meaningful hierarchy level must be at least 2 clusters, got {hierarchy[0]}" 
        sorted_hierarchy = sorted(hierarchy)
        assert hierarchy == sorted_hierarchy and len(hierarchy) == len(np.unique(hierarchy).tolist()), f"Hierarchy must be a strictly increasing sequence of cluster numbers."

    def make_combinations(self):
        """
        to build all possible cluster combinations for each level
        """
        n_levels = len(self.hierarchy_components)
        combinations = []
        for i in range(n_levels):
            cluster_combinations = list(product(*(list(range(n_c)) 
                                                for n_c in self.hierarchy_components[:i+1])))
            cluster_combinations = [list(el) 
                                    for el in cluster_combinations]
            combinations.append(cluster_combinations)
        return combinations
    
    def compute_encoding_params(self, x, level):
        tensor_list = []
        for param_encoder in self.encoder[level]:
            tensor_list.append(param_encoder(x))
        return torch.cat(tensor_list, dim=1)
    
    def compute_decoding_params(self, x):
        tensor_list = []
        for param_decoder in self.decoder:
            tensor_list.append(param_decoder(x))
        return torch.cat(tensor_list, dim=1)

    def encode(self, x, at_level=None):
        if at_level is None:
            at_level = self.n_levels - 1
        else:
            assert at_level >= 0 and isinstance(at_level, int), f"at_level must be an index, i.e, a positive int but got {at_level}"

        HIERARCHY_COMPONENTS = self.hierarchy_components[:at_level+1] # ensure we do not waste time computing unecessary levels
        BATCH_SIZE = x.size(0)
        # build all n_components one-hots once, 
        # then expand to batch
        all_eyek = [torch.eye(n_components, device=x.device) 
                    for n_components in HIERARCHY_COMPONENTS]
        all_z = [] # list of list
        all_latent = []
        all_z_mix = [] # list of tensors
        all_latent_mix = []
        previous_probas = []
        all_cross_level_probas = []
        # For each cluster we need to compute the z (mixture)
        for level, n_components in enumerate(HIERARCHY_COMPONENTS):
            if level == 0:
                # We approximate p(y_1) by q_(y_1 | x)
                # B x n_components
                cluster_probas_level = self.clustering_block[level](x)
            else:
                # We approximate p(y_l | y1, ..., y_l-1) by q_(y_l | x, ..., y_l-1)
                
                # add input features and higher level cluster probas 
                # B x (input_dim + n_components_1 + n_components_2 etc..)
                cluster_probas_in = torch.cat([x] + previous_probas, dim=1) 
                # B x n_components
                cluster_probas_level = self.clustering_block[level](cluster_probas_in)
            
            # register cluster probas for this level
            previous_probas.append(cluster_probas_level)

            # register joint probabilities across levels
            cross_level_probas = torch.clone(cluster_probas_level)
            for i in range(1,level+1):
                previous = torch.clone(previous_probas[level-i])
                for _ in range(i):
                    previous = previous.unsqueeze(1)
                cross_level_probas = cross_level_probas.unsqueeze(-1) * previous
            cross_level_probas = torch.flatten(cross_level_probas, start_dim=1)
            all_cross_level_probas.append(cross_level_probas)

            all_z_level = []
            all_latent_level = []
            level_combinations = self.combinations[level]
            n_combintations = len(level_combinations)
            cross_combinations_all_ck = [[all_eyek[level][k].expand(x.size(0), -1) 
                                        for level,k in enumerate(combinations)]
                                        for combinations in level_combinations]
            cross_combinations_enc_in = [torch.cat([x] + all_c_k, dim=1) 
                                         for all_c_k in cross_combinations_all_ck]
            cross_combinations_enc_in = torch.cat(cross_combinations_enc_in, dim=0)
            # get latent dist parameters
            cross_combinations_latent_k = self.all_posterior_latent[level].constraints(self.compute_encoding_params(cross_combinations_enc_in, level))
            for i in range(n_combintations):
                all_latent_level.append(cross_combinations_latent_k[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]) # params for q_k(z|x,y1=k1,y2=k2,...,y_l=kl)

            # sample 1 time for each sample in the batch
            # = get the latent sample under the condition y = (k1,k2,...,kl)
            cross_combinations_z_k = self.all_posterior_latent[level].sample(cross_combinations_latent_k, 
                                                                             BATCH_SIZE*n_combintations)
            for i in range(n_combintations):
                all_z_level.append(cross_combinations_z_k[i*BATCH_SIZE:(i+1)*BATCH_SIZE,:]) # sample from q_k
            
            all_z.append(all_z_level)
            all_latent.append(all_latent_level)

            z_mixture = torch.stack([t.unsqueeze(1) for t in all_z_level], dim=1).squeeze() # B x K x latent_dim
            z = (cross_level_probas.unsqueeze(-1) * z_mixture).sum(dim=1) # B x latent_dim
            all_z_mix.append(z)

            # latent_params: same "mixture-averaged" params (rarely needed downstream)
            lp = torch.stack([t.unsqueeze(1) for t in all_latent_level], dim=1).squeeze()  # B x K x (Dz * n_params/posterior)
            latent_params = (cross_level_probas.unsqueeze(-1) * lp).sum(dim=1)
            all_latent_mix.append(latent_params)
        # We output latent, probas and actual hidden variables for the last level, as well as all detailed other
        return (all_z_mix[-1], 
                all_latent_mix[-1], 
                previous_probas[-1], 
                all_z, 
                all_latent, 
                all_z_mix, 
                all_latent_mix, 
                previous_probas, 
                all_cross_level_probas)

    
    def decode(self, z):
        input_params = self.compute_decoding_params(z)
        # enforce constraints on params (required for some distributions)
        input_params = self.prior_input.constraints(input_params)
        return input_params
    
    def forward(self, x, at_level=None):
        (z_mix, 
         latent_params_mix, 
         cluster_probas, 
         all_z, 
         all_latent,
         all_z_mix, 
         all_latent_mix, 
         all_cluster_probas,
         all_cross_level_probas
         ) = self.encode(x, at_level=at_level)

        input_params = self.decode(z_mix)
        return (input_params, 
                z_mix, 
                latent_params_mix, 
                cluster_probas, 
                all_z, 
                all_latent, 
                all_z_mix, 
                all_latent_mix, 
                all_cluster_probas, 
                all_cross_level_probas)
    
    def cluster_input(self, x=None, cluster_probas=None, at_level=None): # AFFECTED
        if x is None:
            assert cluster_probas is not None, f"Missing cluster_probas: When given no input features you must provide precomputed cluster_probas"
        else:
            cluster_probas = self.encode(x, at_level=at_level)[2]
        return torch.argmax(cluster_probas, dim=1)

    def log_likelihood_input(self, x, params):
        return self.prior_input.log_likelihood(x, params)
    
    def kl_div(self, z=None, learned_params=None, at_level=None):
        if at_level is None:
            at_level = self.n_levels - 1
        if z is None:
            assert learned_params is not None, "learned_params required when z is None"
            prior_latent = self.all_prior_latent[at_level]
            prior_params = prior_latent.get_reference_params(learned_params)
            if self.all_posterior_latent[at_level].parametric_kl:
                # KL(q_post || p_prior)
                kl = self.all_posterior_latent[at_level].kl_divergence(learned_params, prior_params)
                # sum over latent dims, keep batch dim
                return kl
            else:
                # Monte Carlo estimation: sample z ~ q 
                # tehn comput log q - log p (average outside)
                B = learned_params.size(0)
                z = self.all_posterior_latent[at_level].sample(learned_params, B)
                log_q = self.all_posterior_latent[at_level].log_likelihood(z, learned_params)
                log_p = prior_latent.log_likelihood(z, prior_params)
                return (log_q - log_p)
        else:
            # Monte Carlo estimate per z
            prior_latent = self.all_prior_latent[at_level]
            prior_params = prior_latent.get_reference_params(z)
            log_q = self.all_posterior_latent[at_level].log_likelihood(z, learned_params)
            log_p = prior_latent.log_likelihood(z, prior_params)
            # (average outside)
            return (log_q - log_p) # because kl loss = Eq[log q_z/p_z] and the average is done outside
    
    def iwae(self, batch_x, N = 500, at_level=None):
        """
        Estimates total marginal log-likelihood using 
        the IWAE from Burga et al. in 2016
        """
        if at_level is None:
            at_level = self.n_levels - 1
        self.eval()
        with torch.no_grad():
            (z_mix, 
            latent_params_mix, 
            cluster_probas, 
            all_z, 
            all_latent,
            all_z_mix, 
            all_latent_mix, 
            all_cluster_probas,
            all_cross_level_probas
            ) = self.encode(batch_x, at_level=at_level)
            
            batch_size = batch_x.size(0)
            clusters = torch.argmax(cluster_probas,dim=1)

            # fetch latent from the appropriate cluster for each input sample
            selected_latents = torch.cat([all_latent[-1][c][i:i+1,:] 
                                        for i,c in enumerate(clusters)], 
                                        dim=0)
            sum_paramdims = selected_latents.size(1)
            latent_params = selected_latents.unsqueeze(1) # batch, 1, sum paramdims
            latent_params = latent_params.expand(batch_size, N, sum_paramdims)
            
            # sample from latent posterior
            posterior_latent = self.all_posterior_latent[at_level]
            sampled_z = posterior_latent.sample(latent_params, batch_size)

            # Decode: parameters of p(x|z,y). Shape: (B*N, sum_x_paramdims)
            z_flat = sampled_z.reshape(batch_size * N, self.latent_dim)
            x_params = self.decode(z_flat) 

            # Align x with decoder params
            # If likelihood is per-feature (e.g. NB for genes), x is (B, G).
            # Repeat each x_i N times to match (B*N, G).
            x_rep = batch_x.unsqueeze(1).expand(batch_size, N, batch_x.size(1)).reshape(batch_size * N, -1)

            # log p(x|z,y): should depend on z through x_params
            log_p_x_zy = self.prior_input.log_likelihood(x_rep, x_params)
            # If log_likelihood returns per-feature contributions, sum over features:
            if log_p_x_zy.ndim > 1:
                log_p_x_zy = log_p_x_zy.sum(dim=1)
            log_p_x_zy = log_p_x_zy.reshape(batch_size, N)

            # log p(z)
            prior_latent = self.all_prior_latent[at_level]
            prior_latent_params = prior_latent.get_reference_params(sampled_z.view(-1, self.latent_dim))
            log_p_z = prior_latent.log_likelihood(sampled_z.view(-1, self.latent_dim), prior_latent_params)
            log_p_z = log_p_z.reshape(batch_size, N)
            
            # log q(z|x,y)
            log_q_z_xy = posterior_latent.log_likelihood(sampled_z.view(-1, self.latent_dim),
                                                        latent_params.reshape(-1, latent_params.size(-1))
                                                        )
            log_q_z_xy = log_q_z_xy.reshape(batch_size, N)

            # Log sum exp trick for stable compute of the likelihood ratios
            logspace_ratio = log_p_x_zy + log_p_z - log_q_z_xy
            sumexp = torch.logsumexp(logspace_ratio, dim=1)
            return sumexp - torch.log(torch.tensor([N], device=sumexp.device)) # IWAE for each sample in batch_x

def compute_level(level_data, reg_marginal: float = 1.0):
    """Used if forking is enabled in the model elbo step"""
    x = level_data["x"]
    BATCH_SIZE = x.size(0)

    model = level_data["model"]
    level = level_data["level"]
    joint_probas = level_data["joint_probas"]       # (B, n_comb)
    level_probas = level_data["level_probas"]       # (B, K_level)

    all_z_level = level_data["all_z_level"]
    all_latent_level = level_data["all_latent_level"]
    level_n_combinations = len(all_z_level)

    # 1) Reconstruction
    all_z_comb = torch.cat([all_z_level[comb] for comb in range(level_n_combinations)], dim=0)
    params_comb = model.decode(all_z_comb)
    log_px = model.log_likelihood_input(x.repeat(level_n_combinations, 1), params_comb)
    log_px = torch.stack([log_px[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :]
                          for i in range(level_n_combinations)], dim=1)  # (B, n_comb, D)
    recon = (joint_probas.unsqueeze(2) * log_px).sum(dim=1).sum(dim=1).mean()

    # 2) Latent KL
    all_latent_comb = torch.cat([all_latent_level[comb] for comb in range(level_n_combinations)], dim=0)
    kl_vals = model.kl_div(z=None, learned_params=all_latent_comb, at_level=level)  # expected (B*n_comb,)
    kl_vals = torch.stack([kl_vals[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                           for i in range(level_n_combinations)], dim=1)           # (B, n_comb)
    kl_z = (joint_probas * kl_vals).sum(dim=1).mean()

    # 3) Cluster KL (per-sample)
    ref = model.all_prior_categorical[level].get_ref_proba()  # (K,) or (1,K)
    ref_b = broadcast_cat_probs(ref, level_probas)            # (B,K)
    kl_pi = (level_probas * (level_probas.clamp_min(1e-12).log()
                             - ref_b.clamp_min(1e-12).log())).sum(dim=1).mean()

    # 4) Marginal usage KL (dataset-level within batch)
    kl_marg = 0.0
    if reg_marginal and reg_marginal > 0:
        # use the *non-broadcast* reference for KL(qbar||ref)
        if isinstance(ref, torch.Tensor):
            ref_probs = ref
        else:
            ref_probs = torch.tensor(ref, device=level_probas.device, dtype=level_probas.dtype)
        kl_marg = kl_marginal_usage(level_probas, ref_probs)

    return recon, kl_z, kl_pi, kl_marg


def elbo_MoMix_step(model: MoMixVAE,
                    x: torch.Tensor,
                    beta_kl: float = 1.0,
                    reg_marginal: float = 1.0,
                    track_clusters: bool = False,
                    jit: bool = False):

    (input_params,
     z_mixture,
     latent_params_mixture,
     cluster_probas,
     all_z,
     all_latent,
     all_z_mix,
     all_latent_mix,
     all_cluster_probas,
     all_cross_level_probas) = model(x)

    clusters = model.cluster_input(cluster_probas=cluster_probas) if track_clusters else None
    BATCH_SIZE = x.size(0)

    if jit:
        futures = []
        for level in range(model.n_levels):
            level_data = dict(
                x=x,
                model=model,
                level=level,
                joint_probas=all_cross_level_probas[level],
                level_probas=all_cluster_probas[level],
                all_z_level=all_z[level],
                all_latent_level=all_latent[level],
            )
            futures.append(torch.jit.fork(compute_level, level_data, reg_marginal))

        results = [torch.jit.wait(f) for f in futures]

        recon = torch.stack([out[0] for out in results]).sum()
        kl_z  = torch.stack([out[1] for out in results]).sum()
        kl_pi = torch.stack([out[2] for out in results]).sum()
        kl_marg = torch.stack([out[3] for out in results]).sum()

    else:
        recon = 0.0
        kl_z = 0.0
        kl_pi = 0.0
        kl_marg = 0.0

        for level in range(model.n_levels):
            joint_probas = all_cross_level_probas[level]  # (B, n_comb)
            level_probas = all_cluster_probas[level]      # (B, K_level)
            level_n_combinations = len(all_z[level])

            # 1) Reconstruction
            all_z_comb = torch.cat([all_z[level][comb] for comb in range(level_n_combinations)], dim=0)
            params_comb = model.decode(all_z_comb)
            log_px = model.log_likelihood_input(x.repeat(level_n_combinations, 1), params_comb)
            log_px = torch.stack([log_px[i*BATCH_SIZE:(i+1)*BATCH_SIZE, :]
                                  for i in range(level_n_combinations)], dim=1)  # (B, n_comb, D)
            recon = recon + (joint_probas.unsqueeze(2) * log_px).sum(dim=1).sum(dim=1).mean()

            # 2) Latent KL
            all_latent_comb = torch.cat([all_latent[level][comb] for comb in range(level_n_combinations)], dim=0)
            kl_vals = model.kl_div(z=None, learned_params=all_latent_comb, at_level=level)
            kl_vals = torch.stack([kl_vals[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                                   for i in range(level_n_combinations)], dim=1)  # (B, n_comb)
            kl_z = kl_z + (joint_probas * kl_vals).sum(dim=1).mean()

            # 3) Cluster KL (per-sample)
            ref = model.all_prior_categorical[level].get_ref_proba()
            ref_b = broadcast_cat_probs(ref, level_probas)
            kl_pi = kl_pi + (level_probas * (level_probas.clamp_min(1e-12).log()
                                             - ref_b.clamp_min(1e-12).log())).sum(dim=1).mean()

            # 4) Marginal usage KL (batch marginal as proxy for dataset marginal)
            if reg_marginal and reg_marginal > 0:
                ref_probs = ref
                if not isinstance(ref_probs, torch.Tensor):
                    ref_probs = torch.tensor(ref_probs, device=level_probas.device,
                                             dtype=level_probas.dtype)
                kl_marg = kl_marg + kl_marginal_usage(level_probas, ref_probs)

    # normalize per level to compare with other variants
    recon   = recon / model.n_levels
    kl_z    = kl_z / model.n_levels
    kl_pi   = kl_pi / model.n_levels
    
    # not for the kl marginal because it is already substantially lower than others in scale

    elbo = recon - beta_kl * (kl_z + kl_pi) - reg_marginal * kl_marg
    loss = -elbo

    return loss, dict(
        recon=recon.detach(),
        kl_latent=kl_z.detach(),
        kl_cluster=kl_pi.detach(),
        kl_marginal=kl_marg.detach(),
    ), clusters


def IWAE(model, batch_x, N = 500):
    """
    Importance Weighted Autoencoder estimator (Burga et al. 2016)
    """
    if isinstance(model, MixtureVAE):
        pass
    elif isinstance(model, ind_MoMVAE):
        pass
    elif isinstance(model, MoMixVAE):
        pass
    else:
        raise ValueError(f"Unsupported model type {type(model)}. Must be either MixtureVAE, ind_MoMVAE, MoMixVAE")
    return model.iwae(batch_x, N)