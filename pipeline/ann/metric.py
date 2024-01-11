import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics

def mape(y_pred, y_true):
  return (np.abs(y_pred - y_true)/y_true).mean() * 100

def mae(y_pred, y_true):
  return np.abs(y_pred - y_true).mean()

def R2_score(y_pred, y_true):
  from sklearn.metrics import r2_score
  return r2_score(y_true, y_pred)

def mse(y_pred, y_true):
  return np.mean((y_pred - y_true)**2)

def rmse(y_pred, y_true):
  return np.sqrt(mse(y_pred, y_true))