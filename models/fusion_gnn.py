import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class FusionGNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout=0.1):
        super(FusionGNN, self).__init__()
        self.conv1 = GCNConv(input_dim * 2, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Use output_dim here
        self.dropout = nn.Dropout(dropout)

    def forward(self, input1, input2, edge_index=None):
        # Concatenate the input embeddings
        input_tensor = torch.cat((input1, input2), dim=1)

        # If no edge_index is provided, create a fully connected graph
        if edge_index is None:
            num_nodes = input_tensor.size(0)
            edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
        edge_index = edge_index.cuda()
        # Apply GCN layers
        x = F.relu(self.conv1(input_tensor, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)

        # Apply the final linear layer to get the output embedding
        output = self.fc(x)
        return output
