import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics
from typing import Optional

def evaluate(
  model:nn.Module,
  criterion:callable,
  data_loader:DataLoader,
  device:str='cpu',
  metrics:Optional[torchmetrics.metric.Metric]=None
) -> float:
  '''evaluate
  
  Args:
      model: model
      criterions: list of criterion functions
      data_loader: data loader
      device: device
      metrcis: metrics
  '''
  model.eval()
  with torch.inference_mode():
      X, y = next(iter(data_loader))
      X, y = X.to(device), y.to(device)
      output = model(X)
      loss = criterion(output, y)
      if metrics is not None:
        metrics.update(output, y)

  return loss.item()
