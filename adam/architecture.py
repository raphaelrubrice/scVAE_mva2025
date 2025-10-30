import torch
from my_loss import GMMVAE_loss


class GMMVAE(torch.nn.Module):
    """
    A faire:
        - Ajouter des autres lois, seulement la poisson a été faite.
        - Ajouter l'échantillonnage de plusieurs z, ici un seul est tiré.
        - Ajouter la possibilité de forcer y (ie ne pas l'apprendre) ici y est apprit.
        - Ajouter la possibilité de mettre une autre loi sur z, ici c'est une GMM.
    """

    def __init__(self, N: int, M: int, L: int, K: int, x_law: str) -> None:
        """
        Initialisation du modèle:
            - N (int) = Le nombre de point.
            - M (int) = Le nombre de dimension d'un point.
            - L (int) = La dimension la variable latente.
            - K (int) = Le nombre de classe à apprendre pour la GMM.
            - x_law (str) = La loi de probabilité de x, sert au calcul des paramètres de x.
        """
        super(GMMVAE, self).__init__()

        self.N, self.M, self.L, self.K, self.x_law = N, M, L, K, x_law

        # Prend [x;y] et calcul µ, permet d'avoir q(z | x, y).
        self.f_z_mu = torch.nn.Sequential(
            torch.nn.Linear(in_features=M+K, out_features=L),
            torch.nn.Identity()
        )

        # Prend [x;y] et calcul s², permet d'avoir q(z | x, y) ~ p(z | y).
        self.f_z_var = torch.nn.Sequential(
            torch.nn.Linear(in_features=M+K, out_features=L),
            torch.nn.ReLU(inplace=False)
        )

        # Prend x et calcul pi, permet d'avoir q(y | x) ~ p(y).
        self.f_y_parameters = torch.nn.Sequential(
            torch.nn.Linear(in_features=M, out_features=K),
            torch.nn.LogSoftmax(dim=-1)
        )

        if x_law in ["P"]:
            # Prend z dépendant de y et calcul lambdas, permet d'avoir p(x | z).
            self.f_x_parameters = torch.nn.Sequential(
                torch.nn.Linear(in_features=L, out_features=M),
                torch.nn.ReLU(inplace=False)
            )

        # Finalement on obtient ~ p(x | z).p(z | y).p(y) pour chaque y et chaque z.

        return None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """
        Calcul de l'inférence des points par batch:
            - x (torch.Tensor) = données sous forme de batch.
        
        return:
            - LAMBDAs (torch.Tensor) = Les paramètres de Poisson pour chaque exemple conditionnées par z = p(x | z)
            - MUs (torch.Tensor) = Les paramètres µ de la loi normale pour chaque z conditionnées par y et x. = q(z | x, y) ~ p(z | y)
            - VARs (torch.Tensor) = Les paramètres s² de la loi normale pour chaque z conditionnées par y et x. = q(z | x, y) ~ p(z | y)
            - z (torch.Tensor) = Les variables latentes samplées par la reparamétrization trick.
            - PIs (torch.Tensor) = Les paramètres pi **en log** de la loi multinomiale de y conditionnées par x. = log(q(y | x)) ~ log(p(y))
        
        dim(x) = (B, N, M)
        dim(y) = (K)
        dim(x_concat_y) = (B, K, N, M+K)
        dim(epsilon) = (B, K, N, L)
        dim(MUs) = (B, K, N, L)
        dim(VARs) = (B, K, N, L)
        dim(z) = (B, K, N, L)
        dim(LAMBDAs) = (B, K, N, M)

        Avec:
            - B: Le nombre de batch.
            - N: Le nombre d'exemple.
            - M: La taille des exemples.
            - K: Le nombre de clusters.
            - L: La dimension de la variable latente.
        """

        # concaténe un vecteur booléen pour chaque x, ie [B, K, N, M+K] avec [0, 0, ..., 1, ..., 0] un 1 à la position k pour tout k=1...K.
        x_concat_y = torch.concat([torch.concat([x[:, None, :, :], y.view(1, 1, K).expand(x.shape[0], self.N, self.K)[:, None, :, :]], dim=-1) for y in torch.eye(self.K)], dim=1)
        
        # bruit gaussien sur chaque dimension, pour chaque exemple, pour chaque classe et sur chaque batch.
        epsilon = torch.randn(x.shape[0], self.K, self.N, self.L)

        # projection et non-linéarité pour le calcul des paramètres.
        PIs = self.f_y_parameters(x)
        MUs, VARs = self.f_z_mu(x_concat_y), self.f_z_var(x_concat_y)

        # reparamétrization trick, rend le sampling différentiable.
        z = MUs + torch.sqrt(VARs)*epsilon

        if self.x_law in ["P"]:
            # projection et non-linéarité pour le calcul des paramètres.
            LAMBDAs = self.f_x_parameters(z)

        return LAMBDAs, MUs, VARs, z, PIs


B, N, M, L, K = 5, 100, 200, 10, 7
RNA_seq = torch.randint(low=0, high=1000, size=(B, N, M), dtype=torch.float)

NN = GMMVAE(N=N, M=M, L=L, K=K, x_law="P")

LAMBDAs, MUs, VARs, z, PIs = NN(RNA_seq)

prior_zGy_mu, prior_zGy_var , prior_y = torch.zeros(size=(K, L)), torch.ones(size=(K, L)), torch.log_softmax(torch.ones(size=(K,))/K, dim=0)
gamma_zGy, gamma_y = 1, 1

loss = GMMVAE_loss(prior_zGy_mu, prior_zGy_var, prior_y, gamma_zGy, gamma_y, "P")

print(loss(RNA_seq, LAMBDAs, MUs, VARs, PIs))