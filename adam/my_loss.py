import torch


class GMMVAE_loss(torch.nn.Module):
    """
    A faire:
        - Ajouter des autres lois, seulement la poisson a été faite.
        - Ajouter un warm-up
    """

    def __init__(self, prior_zGy_mu, prior_zGy_var, prior_y, gamma_zGy, gamma_y):
        """
        Initialisation de la loss:
            - prior_zGy_mu (torch.Tensor) = Tenseur contenant les à priori µ sur chaque dimension sachant y.
            - prior_zGy_var (torch.Tensor) = Tenseur contenant les à priori s² sur chaque dimension sachant y.
            - prior_y (torch.Tensor) = Tenseur contenant les à priori pi.
            - gamma_zGy (float) = Coefficient de pondération pour la KL de z.
            - gamma_y (float) = Coefficient de pondération pour la KL de y.

            
        dim(prior_zGy_mu) = (K, L).
        dim(prior_zGy_var) = (K, L).
        dim(prior_y) = (K).
        dim(gamma_zGy) = (1).
        dim(gamma_y) = (1).
        
        Avec:
            - K: Le nombre de clusters.
            - L: La dimension de la variable latente.
        """

        super(GMMVAE_loss, self).__init__()

        self.prior_zGy_mu, self.prior_zGy_var, self.prior_y, self.gamma_zGy, self.gamma_y = prior_zGy_mu, prior_zGy_var, prior_y, gamma_zGy, gamma_y

    
    def forward(self, x, LAMBDAs, MUs, VARs, PIs):
        """
        dim(x) = (B, N, M)
        dim(y) = (K)
        dim(x_concat_y) = (B, K, N, M+K)
        dim(epsilon) = (B, K, N, L)
        dim(MUs) = (B, K, N, L)
        dim(VARs) = (B, K, N, L)
        dim(z) = (B, K, N, L)
        dim(LAMBDAs) = (B, K, N, M)
        dim(log_xGz) = (B, K, N)
        dim(KL_y) = (1)
        dim(KL_z) = (B, K, N)
        dim(prior_zGy_mu) = (K, L).
        dim(prior_zGy_var) = (K, L).
        dim(prior_y) = (K).
        dim(gamma_zGy) = (1).
        dim(gamma_y) = (1).

        Avec:
            - B: Le nombre de batch.
            - N: Le nombre d'exemple.
            - M: La taille des exemples.
            - K: Le nombre de clusters.
            - L: La dimension de la variable latente.
        """
        log_xGz = torch.sum(LAMBDAs - x[:, None, :, :]*torch.log(LAMBDAs + 1e-10), dim=-1)

        KL_y = torch.nn.functional.kl_div(input=self.prior_y, target=PIs, log_target=True, reduction="batchmean")
        
        KL_z = (1/2)*(
            (VARs/(self.prior_zGy_var[:, None, :] + 1e-10)) + (torch.pow((self.prior_zGy_mu[:, None, :] - MUs), 2)/(self.prior_zGy_var[:, None, :] + 1e-10)) - 1 + 2*torch.log((self.prior_zGy_var[:, None, :]/(VARs + + 1e-10)) + 1e-10)
            )

        KL_z = torch.sum(KL_z, dim=-1)
        
        return torch.mean(torch.sum((log_xGz + self.gamma_zGy*KL_z), dim=1)) + self.gamma_y*KL_y