import torch
import torch.nn.functional as F
import torch.nn as nn

EPS = 1e-6


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z, z_aug):
        """

        :param z: [N, d]
        :param z_aug: [N, M, d]
        :return: loss
        """
        z = F.normalize(z, dim=-1)
        z_aug = F.normalize(z_aug, dim=-1)
        pos_sim = torch.exp(-torch.sum((z_aug - z.unsqueeze(1)) ** 2, dim=-1) / self.temperature)  # [N, M]
        neg_sim = torch.exp(-torch.sum((z.unsqueeze(0) - z.unsqueeze(1))**2, dim=-1) / self.temperature)   # [N, N]
        div = (neg_sim.sum(1) - neg_sim.diag()).unsqueeze(-1) + pos_sim
        loss = -torch.log(pos_sim / (div + EPS))
        return loss.mean()


class ManifoldGlueLoss(nn.Module):
    def __init__(self, geo_regular_coef: float):
        super(ManifoldGlueLoss, self).__init__()
        self.geo_regular_coef = geo_regular_coef

    def forward(self, iso_matrix, log_r_matrix):
        """
        In order of (i,j) (j,k) (i,k)
        :param iso_matrix: Parallel Translation matrix with shape [3, T, M]
        :param log_r_matrix: Log Volume Ratio matrix with shape [2, T]
        :return: geometric persistent loss
        """
        holo_loss = torch.mean(torch.norm(iso_matrix[1] * iso_matrix[0] - iso_matrix[2], p=2, dim=-1) ** 2)
        curv_loss = torch.mean((log_r_matrix[0] - log_r_matrix[1]) ** 2)
        return self.geo_regular_coef * holo_loss, self.geo_regular_coef * curv_loss

