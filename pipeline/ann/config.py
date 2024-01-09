import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from m_nn import MANN
from datetime import datetime
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

config = {

  'files': {
    'X_csv': '../../data/train.csv',
    'y_csv': './trn_y.csv',
    'output_log': datetime.now().strftime("%d%H%M%S"),
    'output_csv': 'five_fold.csv',
  },
  'preprocess_params': {
    'num_idx': 0, # building idx
    'split': 'train', # choose between 'train' and 'test' 
    'tst_size': 24 * 7, # 24 hours x 7 = 7 days
    'scaler': MinMaxScaler(),
    'select_channel_idx':[
                          0, # elec_amount (This must be used)
                          # 1, # temp
                          # 2, # wind_speed
                          # 3, # humidity
                          # 4, # rainfall
                          # 5, # sunshine
                          # 6, # no_elec
                          # 7, # sunlight_have
                          # 8, # rolling_mean
                          ] 
  },
  'predict_mode' : 'short', # choose between 'short' and 'long' 
  'model': MANN,
  'model_params': {
    'input_dim': 'auto', 
    'output_dim': 'auto',
    'hidden_dim':512,
    'c_n': 'auto',
    'activation': F.relu,
  },

  'train_params': {
    'dataset_params':{
      'window_size': 24 * 3,
      'prediction_size': 24 * 1
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