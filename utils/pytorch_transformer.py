import torch
import torch.nn as nn

def label2onehot(input_tensor, dim):
    """
    Converts a binary tensor of shape (batch_size, points, 1) to a one-hot tensor of shape (batch_size, points, 2),
    mapping 0 to [1, 0] and 1 to [0, 1].
    """
    # Remove the last dimension
    input_tensor = input_tensor.squeeze(-1)  # Shape: (batch_size, points)

    # Ensure the tensor is of integer type
    input_tensor = input_tensor.long()

    # One-hot encode
    one_hot = torch.nn.functional.one_hot(input_tensor, num_classes=dim)  # Shape: (batch_size, points, 2)

    return one_hot

class PytorchTransformer(nn.Module):
    def __init__(self, i_dimensions, h_dimensions, o_dimensions, num_layers=6, num_heads=8, dropout=0.1, max_seq_length=5000):
        super(PytorchTransformer, self).__init__()
        self.i_dimensions = i_dimensions
        self.h_dimensions = h_dimensions
        self.o_dimensions = o_dimensions
        self.max_seq_length = max_seq_length

        # Input embedding layer to project input vector to hidden dimension h_dimensions
        self.input_embedding = nn.Linear(i_dimensions, h_dimensions)

        # Trainable positional embedding
        self.positional_embedding = nn.Embedding(max_seq_length, h_dimensions)

        # Transformer encoder layers (acting as decoder)
        encoder_layer = nn.TransformerEncoderLayer(d_model=h_dimensions, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer to dimension o_dimensions
        self.output_layer = nn.Linear(h_dimensions, o_dimensions)
    
    @staticmethod
    def _combine2(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        '''
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        '''
        # zs = torch.stack((torch.cat([xs_b,                                               # x
        #                              torch.zeros([bsize, points, 1], device=ys_b.device) # y
        #                              ], dim=2), # x
        #                   torch.cat([torch.zeros_like(xs_b, device=ys_b.device),         # x
        #                              ys_b.view(bsize, points, 1)                         # y
        #                              ], dim=2)
        #                   ), dim=2)
        #print('forward')
        #print(xs_b.shape)
        #print(ys_b.shape)
        zs = torch.stack((torch.cat([torch.zeros([bsize, points, 2], device=ys_b.device), # y
                                     xs_b,                                               # x
                                     ], dim=2), # x
                          torch.cat([binary2onehot(ys_b),                                # y
                                     torch.zeros_like(xs_b, device=ys_b.device)          # x
                                     ], dim=2)
                          ), dim=2)

        zs = zs.view(bsize, 2 * points, dim+2)
        #print(zs.shape)
        return zs

    def forward(self, hxy, inds=None):
        zs = label2onehot(hxy, self.i_dimensions).float()
        # Project input to hidden dimension
        zs = self.input_embedding(zs)  # Shape: (batch_size, seq_length, h_dimensions)

        batch_size, seq_length, _ = zs.size()

        # Generate position indices and get positional embeddings
        positions = torch.arange(0, seq_length, device=zs.device).unsqueeze(0).expand(batch_size, seq_length)
        pos_embed = self.positional_embedding(positions)  # Shape: (batch_size, seq_length, h_dimensions)

        # Add positional embeddings to input embeddings
        zs = zs + pos_embed

        # Prepare mask for causal attention
        mask = generate_causal_mask(seq_length, zs.device)  # Shape: (seq_length, seq_length)

        # Transformer expects input of shape (seq_length, batch_size, h_dimensions)
        zs = zs.transpose(0, 1)  # Shape: (seq_length, batch_size, h_dimensions)

        # Apply transformer encoder layers with causal mask
        output = self.transformer_encoder(zs, mask=mask)

        # Transpose back to (batch_size, seq_length, h_dimensions)
        output = output.transpose(0, 1)

        # Project back to original dimension K
        output = self.output_layer(output)  # Shape: (batch_size, seq_length, K)

        return output


    def forward2(self, xs, ys):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, i_dimensions)

        Returns:
            Output tensor of shape (batch_size, seq_length, o_dimensions)
        """

        zs = self._combine2(xs, ys)

        batch_size, seq_length, _ = zs.size()

        # Project input to hidden dimension
        zs = self.input_embedding(zs)  # Shape: (batch_size, seq_length, h_dimensions)

        # Generate position indices and get positional embeddings
        positions = torch.arange(0, seq_length, device=zs.device).unsqueeze(0).expand(batch_size, seq_length)
        pos_embed = self.positional_embedding(positions)  # Shape: (batch_size, seq_length, h_dimensions)

        # Add positional embeddings to input embeddings
        zs = zs + pos_embed

        # Prepare mask for causal attention
        mask = generate_causal_mask(seq_length, zs.device)  # Shape: (seq_length, seq_length)

        # Transformer expects input of shape (seq_length, batch_size, h_dimensions)
        zs = zs.transpose(0, 1)  # Shape: (seq_length, batch_size, h_dimensions)

        # Apply transformer encoder layers with causal mask
        output = self.transformer_encoder(zs, mask=mask)

        # Transpose back to (batch_size, seq_length, h_dimensions)
        output = output.transpose(0, 1)

        # Project back to original dimension K
        output = self.output_layer(output)  # Shape: (batch_size, seq_length, K)

        return output[:, 0::2, :]

def generate_causal_mask(seq_length, device):
    """
    Generates a causal mask to prevent attention to future positions.

    Args:
        seq_length: Length of the input sequence.
        device: The device (CPU or GPU) on which to create the mask.

    Returns:
        A (seq_length, seq_length) boolean tensor where True indicates positions to be masked.
    """
    mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1).bool()
    return mask  # Shape: (seq_length, seq_length)
