"""Sub-module to define the model classes of Mixture VAEs"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
                 norm_layer: callable = nn.BatchNorm1d):
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

        # +1 because n_layers is the number of hidden
        for i in range(self.n_layers+1):
            if i == 0:
                if self.dropout > 0:
                    module = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                    self.act_func,
                                    nn.Dropout(self.dropout),
                                    self.norm_layer(self.hidden_dim))
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
                                    nn.Dropout(self.dropout),
                                    self.norm_layer(self.hidden_dim))
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
                 norm_layer: callable = nn.BatchNorm1d,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_components = n_components
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
                                           final_act_func=nn.Softmax(), # probabilities
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
            param_encoder = BaseBlock(input_dim=self.latent_dim,
                                    hidden_dim=self.hidden_dim,
                                    output_dim=param_dim,
                                    n_layers=self.n_layers,
                                    act_func=self.act_func,
                                    dropout=self.dropout,
                                    norm_layer=self.norm_layer)
            self.decoder.append(param_encoder)
    
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
    
    def encode(self, x):
        # B x n_components
        cluster_probas = self.clustering_block(x)

        # build all n_components one-hots once, 
        # then expand to batch
        eyeK = torch.eye(self.n_components, device=x.device)
        all_z = []
        all_latent = []
        # For each cluster we need to compute the z (mixture)
        for k in range(self.n_components):
            # one hot encoding for the whole batch
            c_k = eyeK[k].expand(x.size(0), -1) # B x n_components
            
            # add input features
            enc_in = torch.cat([x, c_k], dim=1) # B x (input_dim + n_components)

            # get latent dist parameters
            latent_k = self.posterior_latent.constraints(self.compute_encoding_params(enc_in))
            all_latent.append(latent_k) # params for q_k(z|x,c=k)

            # sample 1 time for each sample in the batch
            # = get the latent sample under the condition y = k
            z_k = self.posterior_latent.sample(latent_k, x.size(0))
            all_z.append(z_k) # sample from q_k

        z_mixture = torch.stack(all_z, dim=1) # B x K x latent_dim
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
        
def elbo_mixture_step(model, x, beta_kl=1.0):
    # Forward (keeps all component samples)
    (input_params, 
     z_mixture, 
     latent_params_mixture, 
     cluster_probas, 
     all_z, 
     all_latent
     ) = model(x)

    B = x.size(0)
    K = model.n_components

    # see supplementary 1
    # 1) Reconstruction: sum_k pi_k * E_{q_k} [ log p(x|z) ]
    # One Monte carlo sample per component: z_k in all_z[k]
    log_px_per_k = []
    for k in range(K):
        # params of p(x|z_k)
        params_k = model.decode(all_z[k]) 
        # compute log p(x|z_k)
        log_px_k = model.prior_input.log_likelihood(x, params_k) 
        # sum over features inside = shape B
        log_px_per_k.append(log_px_k)
    log_px_per_k = torch.stack(log_px_per_k, dim=1) # [B, K]
    recon = (cluster_probas.unsqueeze(2) * log_px_per_k).sum(dim=1).mean() # scalar
    
    # 2) Latent KL: sum_k pi_k * KL(q_k || p)
    kl_per_k = []
    for k in range(K):
        # expects B after summing over latent dims internally
        kl_k = model.kl_div(z=None, learned_params=all_latent[k])
        # if kl_k has extra dims, reduce over latent dims here
        if kl_k.dim() > 1:
            kl_k = kl_k.sum(dim=1)
        kl_per_k.append(kl_k)
    kl_per_k = torch.stack(kl_per_k, dim=1)                # [B, K]
    kl_z = (cluster_probas * kl_per_k).sum(dim=1).mean()               # scalar

    # 3) Cluster KL: KL(π(x) || p(c))
    ref = model.prior_categorical.get_ref_proba() # scalar or [K]
    ref = ref.to(cluster_probas.device)
    ref = ref.expand_as(cluster_probas) if ref.ndim == 1 else ref
    kl_pi = (cluster_probas * (cluster_probas.clamp_min(1e-12).log() - ref.clamp_min(1e-12).log())).sum(dim=1).mean()

    elbo = recon - beta_kl * (kl_z + kl_pi)
    loss = -elbo
    return loss, dict(recon=recon.detach(), 
                      kl_latent=kl_z.detach(), 
                      kl_cluster=kl_pi.detach())


def summed_elbo_mixture_step(model, x, betas_kl = None):
    """
    
    """
    if betas_kl == None:    betas_kl = [1 for _ in range(len(model.branches))]

    P = {"recon":0,
        "kl_latent":0,
        "kl_cluster":0}
    
    L  = 0

    for m, beta in zip(model.branches, betas_kl):
        loss_value, parts = elbo_mixture_step(m, x, beta)

        L += (loss_value/len(model.branches))

        for pname in parts:
            P[pname] += (parts[pname]/len(model.branches))

    return L, P

class ind_MoMVAE(nn.Module):
    """
    
    """
    
    def __init__(self, PARAMS: list[dict[str, float]]) -> None:
        """
        
        """
        super(ind_MoMVAE, self).__init__()
        self.branches = nn.ModuleList([MixtureVAE(**params) for params in PARAMS])

        return None
    

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor]]:
        """
        
        """
        return [MVAE(x) for MVAE in self.branches]


class MoMVAE(nn.Module):
    """
    Implementing a Mixture of Mixtures VAE as a framework
    for built in hierarchical clustering.
    (MoMVAE experimental)
    """
    # TODO
    #  believe most things stay the same 
    # main change is that instead of going through each k once
    # we need to go through each k for each previous k
    # this gets huge quickly ex: for a hierarchical clustering 2, 6, 9
    # we have 3 level of hierarchy, each has a number of clusters
    # we first need to compute the probas (PIs or cluster_probas) 
    # for the highest level (2) then use x, pi1 to compute the next level (6)
    # and finally we use x, pi1, pi2 to compute the final level (9)
    # then we do a 2 x 6 x 9 loop to compute all possible latent params and samples
    # the average z outputed will then be the average over all hierarchies !
    # same scheme in the elbo 