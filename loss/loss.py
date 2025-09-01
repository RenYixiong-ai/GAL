import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def FB_smi_distance(sample, labels, re_grad=True):
    """Compute Fermi/Bose distances between all sample pairs.

    Parameters
    ----------
    sample : torch.Tensor
        Activations from a network layer with shape ``(batch, d)`` or higher.
    labels : torch.Tensor
        One-hot encoded labels corresponding to ``sample``.
    re_grad : bool, optional
        If ``False``, gradients are detached from the inputs.
    """

    sample = sample.view(sample.shape[0], -1)  # flatten to 2D tensor
    batch, hidden_dim = sample.shape

    if not re_grad:
        sample = sample.detach()
        labels = labels.detach()

    # Broadcast to compute pairwise squared L2 distances
    sample_diff = sample.unsqueeze(1) - sample.unsqueeze(0)  # (batch, batch, d)
    D_matrix = torch.sum(sample_diff**2, dim=2) / hidden_dim

    # Compute label similarity matrix (batch, batch)
    label_matrix = labels @ labels.T

    # Number of intra-class (boson) and inter-class (fermi) pairs
    num_boson = (label_matrix.sum() - batch).detach() / 2.0
    num_fermi = (batch**2 - batch - 2 * num_boson).detach() / 2.0

    # Compute boson and fermi distances
    bose_loss = torch.triu(D_matrix * label_matrix, diagonal=1)
    dis_boson = bose_loss.sum() / num_boson + 1e-5
    fermi_loss = torch.triu(D_matrix * (1 - label_matrix), diagonal=1)
    dis_fermi = fermi_loss.sum() / num_fermi

    return dis_fermi, dis_boson


class FDBLoss(nn.Module):
    """Geometry-aware loss enforcing a Fermi/Bose distance ratio."""

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, inputs, targets):
        targets = F.one_hot(targets).float()  # encode labels
        dis_fermi, dis_boson = FB_smi_distance(inputs, targets)
        return torch.abs(dis_fermi / dis_boson - self.alpha)


if __name__ == "__main__":
    pass

