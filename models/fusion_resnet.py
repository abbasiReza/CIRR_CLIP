import torch
import torch.nn as nn

class FusionResNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[768, 768], dropout=0.1):
        super(FusionResNet, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input1, input2):
        input_tensor = torch.cat((input1, input2), dim=1)
        residual = input_tensor
        x = self.relu(self.fc1(input_tensor))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x += input1
        output = self.fc3(x)
        return output
