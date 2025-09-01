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

def train_with_readout(fixed_network, target_network, readout_head, data_loader, optimizer, criterion_fbm, criterion_cross, beta, device):
    if fixed_network is not None:
        fixed_network.eval()    # 固定网络不训练
    target_network.train()      # 目标网络训练
    total_loss = 0
    total_loss_FBM = 0
    total_loss_cross = 0

    for inputs, labels in data_loader:
        #inputs = inputs.view(inputs.shape[0], -1)  # 将图像展平
        inputs, labels = inputs.to(device), labels.to(device)

        # 如果固定网络不为空，数据先通过固定网络（不计算梯度）
        outputs = inputs
        if fixed_network is not None:
            with torch.no_grad():
                outputs = fixed_network(inputs)

        # 数据通过目标网络
        target_outputs = target_network(outputs)

        # 计算FBM损失
        loss_FBM = criterion_fbm(target_outputs, labels)
        total_loss_FBM += loss_FBM.item()

        # 数据通过读出头网络
        logits = readout_head(target_outputs)

        # 计算交叉熵损失
        loss_cross = criterion_cross(logits, labels)
        total_loss_cross += loss_cross.item()

        # 总损失
        loss = loss_FBM + beta * loss_cross
        total_loss += loss.item()

        # 反向传播优化目标网络
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    return total_loss / len(data_loader), total_loss_FBM / len(data_loader), total_loss_cross / len(data_loader)


if __name__ == "__main__":
    pass


