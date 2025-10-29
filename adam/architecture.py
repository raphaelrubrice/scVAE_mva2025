import torch


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

        PI = self.f_y_parameters(x)

        MU, VAR = self.f_z_mu(x_concat_y), self.f_z_var(x_concat_y)

        z = MU + torch.sqrt(VAR)*epsilon

        LAMBDA = self.f_x_parameters(z)

        return LAMBDA, z, PI


N, M = 100, 200
RNA_seq = torch.randint(low=0, high=1000, size=(N, M), dtype=torch.float)

NN = GMMVAE(N=100, M=200, L=10, K=5)

LAMBDA, z, PI = NN(RNA_seq)

print(LAMBDA.size())
print(z.size())
print(PI.size())