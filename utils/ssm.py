import torch
import torch.nn as nn
from mamba_ssm import Mamba
from utils.models import label2onehot
class KLayerMambaModel(nn.Module):
    def __init__(self, num_tokens, num_layer, d_model, d_state, d_conv, expand):
        """
        Initializes a Mamba model with k layers.

        Args:
            num_layer (int): Number of Mamba layers.
            d_model (int): Dimension of the model (embedding size).
            d_state (int): SSM state expansion factor.
            d_conv (int): Local convolution width.
            expand (int): Block expansion factor.
        """
        super(KLayerMambaModel, self).__init__()
        self.num_tokens = num_tokens
        self.num_layer = num_layer
        self.i_embedding = nn.Linear(num_tokens, d_model)
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            for _ in range(num_layer)
        ])
        self.o_embedding = nn.Linear(d_model, num_tokens)
        # Optional: Add Layer Normalization after each layer
        #self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Forward pass through the k-layer Mamba model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, length, d_model).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        x = label2onehot(x, self.num_tokens).float()
        x = self.i_embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.o_embedding(x)
        return x