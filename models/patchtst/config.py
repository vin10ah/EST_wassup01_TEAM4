import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from patchtst import PatchTST
from datetime import datetime
import pytz

config = {

  'files': {
    'X_trn': './data/X_trn.csv',
    'y_trn': './data/y_trn.csv',
    'X_tst': './data/X_tst.csv',
    'y_tst': './data/y_tst.csv',
    'output_log': datetime.now(pytz.timezone('Asia/Seoul')).strftime("%d%H%M%S"),
    'output_csv': './results/five_fold.csv',
  },

  'model': PatchTST,
  'model_params': {
    'n_token': 8,
    'input_dim': 16, # input_dim(patch_length) Must be an even number
    'model_dim': 512,
    'num_heads': 8,
    'num_layers': 4,
    'output_dim': 4,
  },

  'train_params': {
    'dataset_params':{
      'patch_length': 16,
      'n_patches': 8
    },
    'trn_data_loader_params': {
      'batch_size': 128,
      'shuffle': True
    },
    'tst_data_loader_params': {
      'batch_size': 'auto',
      'shuffle': False
    },
    'loss_fn': F.mse_loss,
    'optim': torch.optim.AdamW,
    'optim_params': {
      'lr': 0.001,
      'weight_decay': 0
    },
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

  'cv_params':{
    'n_splits': 5,
    'shuffle':False,
  }
}