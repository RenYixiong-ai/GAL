from loss.loss import FDBLoss

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from utils import *

def weighted_balanced_loss(L1, L2, beta):
    """
    L1: structure loss (e.g., small ~0.02)
    L2: cross-entropy loss (e.g., large ~0.7)
    beta: desired proportion of L2's contribution in [0,1]
    """
    if L1.item() == 0:
        alpha = 0.0  # Avoid division by zero
    else:
        alpha = (1 - beta) * L2.item() / L1.item()
    
    loss_total = alpha * L1 + beta * L2
    return loss_total

def train_with_readout(
    fixed_network,
    target_network,
    readout_head,
    data_loader,
    optimizer,
    criterion_fbm,
    criterion_cross,
    beta,
    device,
):
    """Train a single layer with an auxiliary readout.

    Parameters
    ----------
    fixed_network : nn.Module or None
        Previously trained part of the network whose parameters remain frozen.
    target_network : nn.Module
        The layer currently being trained.
    readout_head : nn.Module
        Temporary linear classifier used to supply a supervised signal.
    data_loader : DataLoader
        Iterator yielding training batches.
    optimizer : torch.optim.Optimizer
        Optimizer for the ``target_network`` parameters.
    criterion_fbm : callable
        Geometry-aware loss function (FDBLoss).
    criterion_cross : callable
        Cross-entropy loss for the readout.
    beta : float
        Weight of the cross-entropy term.
    device : torch.device
        Device on which computation takes place.
    """

    if fixed_network is not None:
        fixed_network.eval()    # keep the previous network frozen
    target_network.train()      # train the current target network
    total_loss = 0
    total_loss_FBM = 0
    total_loss_cross = 0

    for inputs, labels in data_loader:
        #inputs = inputs.view(inputs.shape[0], -1)  # flatten images if needed
        inputs, labels = inputs.to(device), labels.to(device)

        # If a fixed network exists, forward data through it without gradients
        outputs = inputs
        if fixed_network is not None:
            with torch.no_grad():
                outputs = fixed_network(inputs)

        # Forward through the target network
        target_outputs = target_network(outputs)

        # Compute FBM loss
        loss_FBM = criterion_fbm(target_outputs, labels)
        total_loss_FBM += loss_FBM.item()

        # Forward through the readout head
        logits = readout_head(target_outputs)

        # Compute cross-entropy loss
        loss_cross = criterion_cross(logits, labels)
        total_loss_cross += loss_cross.item()

        # Total loss
        loss = loss_FBM + beta * loss_cross
        total_loss += loss.item()

        # Backpropagate and update the target network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (
        total_loss / len(data_loader),
        total_loss_FBM / len(data_loader),
        total_loss_cross / len(data_loader),
    )


if __name__ == "__main__":
    pass


