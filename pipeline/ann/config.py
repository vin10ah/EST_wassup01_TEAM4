import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from m_nn import MANN
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

config = {

  'files': {
    'data': '../../data/train.csv',
  },
  'output': {
    'output_log': datetime.now().strftime("%d%H%M%S"),
    'load_pth_24':'../../results/ann/best24.pth',
    'load_pth_168':'../../results/ann/best168.pth',
    'load_pth_168d':'../../results/ann/best24d.pth',
  },
  'preprocess_params': {
    'num_idx': 0, # building idx
    'split': 'train', # choose between 'train' and 'test' 
    'tst_size': 24 * 1, # 24 hours x n = n days
    'scaler': MinMaxScaler(),
    'scaler2': MinMaxScaler(),
    'select_channel_idx':[
                          0, # elec_amount (This must be used)
                          1, # temp
                          # 2, # wind_speed
                          # 3, # humidity
                          4, # rainfall
                          # 5, # sunshine
                          6, # rolling_mean
                          7, # diff
                          ]
  },
  'predict_mode' : 'one_step', # choose between 'one_step' and 'dynamic'
  'model': MANN,
  'model_params': {
    'input_dim': 'auto',
    'output_dim': 'auto',
    # 'hidden_dim':512,
    'c_n': 'auto',
    'activation': F.relu,
  },

  'train_params': {
    'dataset_params':{
      'window_size': 24 * 3,
      'prediction_size': 24 * 1
    },
    'trn_data_loader_params': {
      'batch_size': 64,
      'shuffle': True,
    },
    'tst_data_loader_params': {
      'batch_size': 'auto',
      'shuffle': False
    },
    'loss_fn': nn.functional.mse_loss,
    'optim': torch.optim.AdamW,
    'optim_params': {
      'lr': 0.00001,
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
    'epochs': 270,
  },
}