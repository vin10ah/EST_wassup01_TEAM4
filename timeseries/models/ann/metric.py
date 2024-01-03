import numpy as np
import torch
import torch.nn.functional as F

def mape(y_pred, y_true):
  return (np.abs(y_pred - y_true)/y_true).mean() * 100

def mae(y_pred, y_true):
  return np.abs(y_pred - y_true).mean()

def r2_score(y_true, y_pred):
  from sklearn.metrics import r2_score
  return r2_score(y_true, y_pred)

def mse(y_true, y_pred):
  return F.mse_loss(y_true, y_pred, reduction='mean')

def rmse(y_true, y_pred):
  return torch.sqrt(F.mse_loss(y_true, y_pred, reduction='mean'))