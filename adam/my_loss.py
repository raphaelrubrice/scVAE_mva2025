import torch


class GMMVAE_loss(torch.nn.Module):
    """
    
    """

    def __init__(self, prior_zGy, prior_y, gamma_zGy, gamma_y):
        """
        
        """

        super(GMMVAE_loss, self).__init__()

        self.prior_zGy, self.prior_y, self.gamma_zGy, self.gamma_y = prior_zGy, prior_y, gamma_zGy, gamma_y

    
    def forward(self, x, LAMBDAs, MUs, VARs, PIs):
        """
        
        """
        log_xGz = torch.sum(LAMBDAs - x*torch.log(LAMBDAs + 1e-10), dim=-1)
        print(log_xGz.size())
        return log_xGz