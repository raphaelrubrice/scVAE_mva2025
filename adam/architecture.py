import torch
from my_loss import GMMVAE_loss


class GMMVAE(torch.nn.Module):
    """
    
    """

    def __init__(self, N, M, L, K):
        """
        
        """
        super(GMMVAE, self).__init__()

        self.N, self.M, self.L, self.K = N, M, L, K

        self.f_z_mu = torch.nn.Sequential(
            torch.nn.Linear(in_features=M+K, out_features=L),
            torch.nn.Identity()
        )

        self.f_z_var = torch.nn.Sequential(
            torch.nn.Linear(in_features=M+K, out_features=L),
            torch.nn.ReLU(inplace=False)
        )

        self.f_y_parameters = torch.nn.Sequential(
            torch.nn.Linear(in_features=M, out_features=K),
            torch.nn.Softmax(dim=-1)
        )

        self.f_x_parameters = torch.nn.Sequential(
            torch.nn.Linear(in_features=L, out_features=M),
            torch.nn.ReLU(inplace=False)
        )

    def forward(self, x):
        """
        
        """
        x_concat_y = torch.concat([torch.concat([x, torch.tile(y, (self.N, 1))], dim=-1)[None, :, :] for y in torch.eye(self.K)], dim=0)
        epsilon = torch.randn(self.K, self.N, self.L)

        PIs = self.f_y_parameters(x)

        MUs, VARs = self.f_z_mu(x_concat_y), self.f_z_var(x_concat_y)

        z = MUs + torch.sqrt(VARs)*epsilon

        LAMBDAs = self.f_x_parameters(z)

        return LAMBDAs, MUs, VARs, z, PIs


N, M, L, K = 100, 200, 10, 5
RNA_seq = torch.randint(low=0, high=1000, size=(N, M), dtype=torch.float)

NN = GMMVAE(N=N, M=M, L=L, K=K)

LAMBDAs, MUs, VARs, z, PIs = NN(RNA_seq)

prior_zGy, prior_y = (torch.zeros(size=(K, L)), torch.ones(size=(K, L))), torch.ones(size=(K,))/K
gamma_zGy, gamma_y = 1, 1

loss = GMMVAE_loss(prior_zGy, prior_y, gamma_zGy, gamma_y)

loss(RNA_seq, LAMBDAs, MUs, VARs, PIs)