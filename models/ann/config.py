import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nn import ANN
from datetime import datetime
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError

config = {

  'files': {
    'X_csv': '../../data/train.csv',
    'y_csv': './trn_y.csv',
    'output_log': datetime.now().strftime("%d%H%M%S"),
    'output_csv': 'five_fold.csv',
  },

  'model': ANN,
  'model_params': {
    'input_dim': 'auto', 
    'output_dim': 'auto',
  },

  'train_params': {
    'dataset_params':{
      'window_size': 10,
      'prediction_size': 5
    },
    'trn_data_loader_params': {
      'batch_size': 32,
      'shuffle': True,
    },
    'tst_data_loader_params': {
      'batch_size': 'auto',
      'shuffle': False
    },
    'loss_fn': nn.functional.mse_loss,
    'optim': torch.optim.AdamW,
    'optim_params': {
      'lr': 0.01,
      'weight_decay': 0
    },
    'metric': MetricCollection({
               'mae': MeanAbsoluteError(),
               'mape':MeanAbsolutePercentageError(),
               }),
    'lr_scheduler': ReduceLROnPlateau,
    'scheduler_params': {
      'mode': 'min',
      'factor': 0.1,
      'patience': 5,
      'verbose':False
    },
    
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'epochs': 1,
  },

}