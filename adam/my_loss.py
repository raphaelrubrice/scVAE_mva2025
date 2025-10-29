import torch


class GMMVAE_loss(torch.nn.Module):
    """
    
    """

    def __init__(self, prior_zGy_mu, prior_zGy_var, prior_y, gamma_zGy, gamma_y):
        """
        
        """

        super(GMMVAE_loss, self).__init__()

        self.prior_zGy_mu, self.prior_zGy_var, self.prior_y, self.gamma_zGy, self.gamma_y = prior_zGy_mu, prior_zGy_var, prior_y, gamma_zGy, gamma_y

    
    def forward(self, x, LAMBDAs, MUs, VARs, PIs):
        """
        
        """
        log_xGz = torch.sum(LAMBDAs - x*torch.log(LAMBDAs + 1e-10), dim=-1)

        KL_y = torch.nn.functional.kl_div(input=self.prior_y, target=PIs, log_target=True, reduction="batchmean")
        
        KL_z = (1/2)*(
            (VARs/(self.prior_zGy_var[:, None, :] + 1e-10)) + (torch.pow((self.prior_zGy_mu[:, None, :] - MUs), 2)/(self.prior_zGy_var[:, None, :] + 1e-10)) - 1 + 2*torch.log((self.prior_zGy_var[:, None, :]/(VARs + + 1e-10)) + 1e-10)
            )
        
        KL_z = torch.sum(KL_z, dim=-1)
        
        return torch.mean(torch.sum((log_xGz + self.gamma_zGy*KL_z), dim=0), dim=0) + self.gamma_y*KL_y