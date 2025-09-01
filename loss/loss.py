import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

import math

# FBM距离
def FB_smi_distance(sample, labels, re_grad=True):
    sample = sample.view(sample.shape[0], -1)  # 将样本展平为二维张量
    batch, hidden1_features = sample.shape

    if not re_grad:
        # 使用 detach 来确保 sample 和 labels 不参与梯度计算
        sample = sample.detach()
        labels = labels.detach()

    # 使用广播计算每对样本之间的 L2 距离的平方和
    sample_diff = sample.unsqueeze(1) - sample.unsqueeze(0)  # 扩展维度并相减，得到 (batch, batch, outdim)
    D_matrix = torch.sum(sample_diff**2, dim=2)/hidden1_features

    # 计算标签矩阵的乘积，结果是 (batch_size, batch_size)
    label_matrix = labels @ labels.T

    # 计算类别的
    num_boson = (label_matrix.sum() - batch).detach()/2.0
    num_fermi = (batch**2 - batch - 2*num_boson).detach()/2.0

    # 计算bose_loss, fermi_loss并归一化
    bose_loss = torch.triu(torch.mul(D_matrix, label_matrix), diagonal=1)
    dis_boson = bose_loss.sum()/num_boson + 1e-5
    fermi_loss = torch.triu(torch.mul(D_matrix, 1-label_matrix), diagonal=1)
    dis_fermi = fermi_loss.sum()/num_fermi

    return dis_fermi, dis_boson

# 定义FBM损失
class FDBLoss(nn.Module):
    def __init__(self, alpha):
        super(FDBLoss, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets):
        targets = F.one_hot(targets).float()  #编码
        dis_fermi, dis_boson = FB_smi_distance(inputs, targets)
        out = torch.abs(dis_fermi/dis_boson - self.alpha)
        return out

if __name__ == "__main__":
    pass