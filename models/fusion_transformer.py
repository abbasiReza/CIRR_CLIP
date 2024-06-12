import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class FusionTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=8, num_layers=2, dropout=0.1):
        super(FusionTransformer, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, input1, input2):
        input_tensor = torch.cat((input1, input2), dim=1).unsqueeze(1)  # Add sequence dimension
        x = self.transformer_encoder(input_tensor)
        x = x.squeeze(1)
        output = self.fc(x)
        return output
