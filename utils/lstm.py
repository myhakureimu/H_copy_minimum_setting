import torch
import torch.nn as nn
from utils.models import label2onehot

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = label2onehot(x, self.input_dim).float()
        #print('x.shape=',x.shape)
        # Initialize hidden and cell states
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim, device=x.device
        )
        c0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim, device=x.device
        )

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Apply the output layer to each time step
        out = self.fc(out)
        return out