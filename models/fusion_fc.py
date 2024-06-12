import torch
import torch.nn as nn

class FusionFC(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[1024, 1024], dropout=0.1):
        super(FusionFC, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, input1, input2):
        input_tensor = torch.cat((input1, input2), dim=1)
        output = self.fc_layers(input_tensor)
        return output
