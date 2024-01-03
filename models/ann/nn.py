import torch.nn as nn

class ANN(nn.Module):
  def __init__(self, input_dim:int=10, output_dim:int=5):
    super().__init__()
    self.linear_activation_stack = nn.Sequential(
        # 1
        nn.Linear(input_dim, 512),
        # nn.BatchNorm1d(512),
        nn.ReLU(),
        
        # 2
        nn.Linear(512, output_dim),
        nn.Sigmoid()
        )
  def forward(self, x):
    logits = self.linear_activation_stack(x)
    return logits