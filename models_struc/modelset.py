import torch.nn as nn
import torch
import torch.nn.functional as F
import copy

#from . import KAN
#from torchvision.models import resnet34, resnet50, resnet18,vgg16,vgg19

"""Model definitions used throughout the project."""


class SingleLayerNetwork(nn.Module):
    """A single fully connected layer with optional layer normalization.

    The module additionally records the contribution of each input feature to
    the output units, which later aids the geometry-aware loss.
    """

    def __init__(self, in_features, out_features, use_layernorm=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.norm = nn.LayerNorm(out_features)
        self.last_mapping = None  # shape: (batch, out_features, in_features + 1)

    def forward(self, x):
        # Weights: (out, in), bias: (out,)
        W = self.linear.weight           # (out_features, in_features)
        b = self.linear.bias             # (out_features,)

        bsz = x.size(0)
        x_expanded = x.unsqueeze(1)             # (batch, 1, in_features)
        W_expanded = W.unsqueeze(0)             # (1, out_features, in_features)
        contrib = x_expanded * W_expanded       # (batch, out_features, in_features)

        # Include bias contribution: broadcast to (batch, out_features)
        b_contrib = b.unsqueeze(0).expand(bsz, -1).unsqueeze(-1)  # (batch, out_features, 1)

        # Concatenate into the final mapping tensor
        self.last_mapping = torch.cat([contrib, b_contrib], dim=-1)  # (batch, out_features, in_features + 1)

        out = self.linear(x)
        if self.use_layernorm:
            out = self.norm(out)

        #return F.relu(self.linear(x))
        return torch.tanh(self.linear(x))
    
    def clone_self(self):
        """Create a detached copy of the layer on the same device."""

        current_device = next(self.parameters()).device

        clone = SingleLayerNetwork(
            in_features=self.linear.in_features,
            out_features=self.linear.out_features,
            use_layernorm=self.use_layernorm,
        )
        clone.load_state_dict(copy.deepcopy(self.state_dict()))
        clone = clone.to(current_device)  # move the model to the same device
        return clone
    
class MultiLayerNetwork(nn.Module):
    """Container that sequentially stores trained layers."""

    def __init__(self):
        super(MultiLayerNetwork, self).__init__()
        self.layers = nn.ModuleList()  # hold layers added one by one

    def add(self, layer):
        """Append a pre-trained layer to the module list."""
        self.layers.append(copy.deepcopy(layer))

    def pop(self):
        """Remove and return the last layer in ``self.layers``."""
        if len(self.layers) > 0:
            last_layer = self.layers[-1]  # get the last layer
            del self.layers[-1]          # delete manually
            return last_layer            # return the removed layer
        else:
            print("Warning: No layers to remove.")
            return None

    def forward(self, x, n_layers=None, return_intermediate=False):
        outputs = []
        mappings = []
        
        # Compute outputs layer by layer
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

class ReadoutHead(nn.Module):
    """Linear readout layer with frozen random weights."""

    def __init__(self, input_size, output_size):
        super(ReadoutHead, self).__init__()
        # Initialize weights with a small Gaussian and freeze parameters
        self.weight = nn.Parameter(
            torch.randn(input_size, output_size) * 0.01, requires_grad=False
        )
        self.bias = nn.Parameter(torch.zeros(output_size), requires_grad=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # Linear transformation: y = xW + b
        return torch.matmul(x, self.weight) + self.bias

if __name__ == "__main__":
    pass
