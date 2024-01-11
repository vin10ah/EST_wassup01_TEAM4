import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from patchtst import PatchTST
from datetime import datetime
import pytz

config = {

  'files': {
    'data': '../../data/train.csv',
    'output_log': datetime.now(pytz.timezone('Asia/Seoul')).strftime("%d%H%M%S"),
  },

  'preprocess_params': {
    'num_idx': 0, # building idx
    'split': 'train', # choose between 'train' and 'test' 
    'tst_size': 24 * 7, # 24 hours x 7 = 7 days
    'scaler': MinMaxScaler(),
    'select_channel_idx':[
                          # 0, # temp
                          # 1, # wind_speed
                          # 2, # humidity
                          # 3, # rainfall
                          # 4, # sunshine
                          # 5, # no_elec
                          # 6, # sunlight_have
                          # 7, # rolling_mean
                          ] 
  },

  'model': PatchTST,
  'model_params': {
    'n_token': 64, # n_patches
    'input_dim': 16, # input_dim(patch_length) Must be an even number
    'model_dim': 128, # node
    'num_heads': 16, # multihead
    'num_layers': 3, # transformer layer
    'output_dim': 24, # prediction_length
  },

  'train_params': {
    'trn_data_loader_params': {
      'batch_size': 64,
      'shuffle': True
    },
    'tst_data_loader_params': {
      'batch_size': 'auto',
      'shuffle': False
    },
    'loss_fn': F.mse_loss,
    'optim': torch.optim.AdamW,
    'optim_params': {
      'lr': 0.0001,
      'weight_decay': 0 # default 0.01
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
}