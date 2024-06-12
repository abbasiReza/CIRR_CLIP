import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionAttention(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout=0.1):
        super(FusionAttention, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, input1, input2):
        input_tensor = torch.cat((input1, input2), dim=1)
        x = F.relu(self.fc1(input_tensor))
        attention_weights = F.softmax(self.attention(x), dim=1)
        x = x * attention_weights
        # x = F.dropout(x, p=self.dropout, training=self.training)
        output = self.fc2(x)
        return output
