import torch.nn as nn
import torch.nn.functional as F

class MANN(nn.Module):
  def __init__(self, input_dim:int=10, output_dim:int=5, hidden_dim:int=512, c_n:int=2, activation=F.elu, dropout:float=0.3):
    super().__init__()
    self.lin1 = nn.Linear(input_dim*c_n, 1024)
    self.batch1 = nn.BatchNorm1d(1024)
    self.lin2 = nn.Linear(1024, 512)
    self.batch2 = nn.BatchNorm1d(512)
    self.lin3 = nn.Linear(512, 256)
    self.batch3 = nn.BatchNorm1d(256)
    self.lin4 = nn.Linear(256, output_dim)
    self.batch4 = nn.BatchNorm1d(1024)
    self.activation = activation
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x = x.flatten(1)    # (B, d_in * c_in)
    x = self.lin1(x)    # (B, d_hidden)
    # x = self.batch1(x)
    x = self.activation(x)
    x = self.dropout(x)
    x = self.lin2(x)    # (B, d_out)
    # x = self.batch2(x)
    x = self.activation(x)
    # x = self.dropout(x)
    x = self.lin3(x)
    # x = self.batch3(x)   # (B, d_out)
    x = self.activation(x)
    # x = self.dropout(x)
    x = self.lin4(x)    # (B, d_out)
    # x = self.batch4(x)
    return F.sigmoid(x)
    