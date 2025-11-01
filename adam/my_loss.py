import torch


class GMMVAE_loss(torch.nn.Module):
    """
    A faire:
        - Ajouter des autres lois, seulement la poisson a été faite.
        - Ajouter un warm-up
    """

    def __init__(self, prior_zGy_mu: torch.Tensor, prior_zGy_var: torch.Tensor, prior_y: torch.Tensor, gamma_zGy: float, gamma_y: float, x_law: str) -> None:
        """
        Initialisation de la loss:
            - prior_zGy_mu (torch.Tensor) = Tenseur contenant les à priori µ sur chaque dimension sachant y.
            - prior_zGy_var (torch.Tensor) = Tenseur contenant les à priori s² sur chaque dimension sachant y.
            - prior_y (torch.Tensor) = Tenseur contenant les à priori pi **en log**.
            - gamma_zGy (float) = Coefficient de pondération pour la KL de z.
            - gamma_y (float) = Coefficient de pondération pour la KL de y.
            - x_law (str) = La loi de probabilité de x, sert au calcul des paramètres de x.

        return:
            - None.

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

        self.prior_zGy_mu, self.prior_zGy_var, self.prior_y, self.gamma_zGy, self.gamma_y, self.x_law = prior_zGy_mu, prior_zGy_var, prior_y, gamma_zGy, gamma_y, x_law

        return None
    
    
    def forward(self, x: torch.Tensor, x_parameters: torch.Tensor, MUs: torch.Tensor, VARs: torch.Tensor, PIs: torch.Tensor) -> torch.Tensor:
        """
        Forward permettant le calcul de la loss.

        Paramètres:
            - x: L'entrée.
            - x_parameters: Les paramètres de la loi de x.
            - MUs: Les µ calculés de chaque z et sur chaque dimensions..
            - VARs Les s² calculés de chaque z et sur chaque dimensions.
            - PIs: Les pis calculés des catégories de y.

        return:
            - (1): La valeur de la loss.

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
        if self.x_law == "P":
            
            LAMBDAs = x_parameters

            # -log probabilité d'une loi de Poisson.
            log_xGz = torch.sum(LAMBDAs - x[:, None, :, :]*torch.log(LAMBDAs + 1e-10), dim=-1)

            

        elif self.x_law == "ZIP":

            LAMBDAs, P = x_parameters

            x_rep = x[:, None, :, :].repeat(1, LAMBDAs.shape[1], 1, 1)

            mask = (x_rep == 0)

            neglog_prob_0 = lambda x: -torch.log(P + (1 - P)*torch.exp(-LAMBDAs) + 1e-10)
            neglog_prob_not_0 = lambda x: -torch.log(1 - P + 1e-10) + LAMBDAs - (x[:, None, :, :]*torch.log(LAMBDAs + 1e-10))

            # -log probabilité d'une loi de zero-inflated-Poisson.
            x_new = torch.where(mask, neglog_prob_0(x), neglog_prob_not_0(x))
            log_xGz = torch.sum(x_new, dim=-1)

        # KL entre deux distributions sous log.
        KL_y = torch.nn.functional.kl_div(input=self.prior_y, target=PIs, log_target=True, reduction="batchmean")
        
        # KL entre deux lois normales. (cf formule)
        KL_z = (1/2)*(
            (VARs/(self.prior_zGy_var[:, None, :] + 1e-10)) + (torch.pow((self.prior_zGy_mu[:, None, :] - MUs), 2)/(self.prior_zGy_var[:, None, :] + 1e-10)) - 1 + 2*torch.log((self.prior_zGy_var[:, None, :]/(VARs + + 1e-10)) + 1e-10)
            )
        KL_z = torch.sum(KL_z, dim=-1)
        
        return torch.mean(torch.sum(PIs.transpose(1, 2)*(log_xGz + self.gamma_zGy*KL_z), dim=1)) + self.gamma_y*KL_y