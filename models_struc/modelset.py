import torch.nn as nn
import torch
import torch.nn.functional as F
import copy

#from . import KAN
#from torchvision.models import resnet34, resnet50, resnet18,vgg16,vgg19

# 定义单层网络
class SingleLayerNetwork(nn.Module):
    def __init__(self, in_features, out_features, use_layernorm=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.norm = nn.LayerNorm(out_features)
        self.last_mapping = None  # shape: (batch, out_features, in_features + 1)

    def forward(self, x):
        # 权重: (out, in), 偏置: (out,)
        W = self.linear.weight           # (out_features, in_features)
        b = self.linear.bias             # (out_features,)

        bsz = x.size(0)
        x_expanded = x.unsqueeze(1)             # (batch, 1, in_features)
        W_expanded = W.unsqueeze(0)             # (1, out_features, in_features)
        contrib = x_expanded * W_expanded       # (batch, out_features, in_features)

        # 加入偏置贡献：广播为 (batch, out_features)
        b_contrib = b.unsqueeze(0).expand(bsz, -1).unsqueeze(-1)  # (batch, out_features, 1)

        # 拼接成最终的映射贡献张量
        self.last_mapping = torch.cat([contrib, b_contrib], dim=-1)  # (batch, out_features, in_features + 1)

        out = self.linear(x)
        if self.use_layernorm:
            out = self.norm(out)

        #return F.relu(self.linear(x))
        return torch.tanh(self.linear(x))
    
    def clone_self(self):
        # 获取当前设备
        current_device = next(self.parameters()).device

        # 创建新模型 + 拷贝参数
        clone = SingleLayerNetwork(
            in_features=self.linear.in_features,
            out_features=self.linear.out_features,
            use_layernorm=self.use_layernorm
        )
        clone.load_state_dict(copy.deepcopy(self.state_dict()))
        clone = clone.to(current_device)  # 将模型移动到相同设备
        return clone
    
# 定义模型
class MultiLayerNetwork(nn.Module):
    def __init__(self):
        super(MultiLayerNetwork, self).__init__()
        self.layers = nn.ModuleList()  # 用于存储逐步添加的网络层

    def add(self, layer):
        # 添加已训练好的网络层到ModuleList中
        self.layers.append(copy.deepcopy(layer))

    def pop(self):
        """删除 self.layers 中的最后一个网络层"""
        if len(self.layers) > 0:
            last_layer = self.layers[-1]  # 获取最后一层
            del self.layers[-1]  # 手动删除
            return last_layer  # 返回被删除的层
        else:
            print("Warning: No layers to remove.")
            return None

    def forward(self, x, n_layers=None, return_intermediate=False):
        outputs = []
        mappings = []
        
        # 逐层计算输出
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if return_intermediate and (n_layers is None or i < n_layers):
                outputs.append(x)
                if isinstance(layer, SingleLayerNetwork):
                    mappings.append(layer.last_mapping)
            if i == n_layers:
                break
        
        if return_intermediate:
            return outputs, mappings
        else:
            return x

# 定义读出头网络
class ReadoutHead(nn.Module):
    def __init__(self, input_size, output_size):
        super(ReadoutHead, self).__init__()
        # 初始化权重为高斯分布，且权重不可训练
        self.weight = nn.Parameter(torch.randn(input_size, output_size) * 0.01, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(output_size), requires_grad=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # 线性变换：y = xW + b
        return torch.matmul(x, self.weight) + self.bias

if __name__ == "__main__":
    pass