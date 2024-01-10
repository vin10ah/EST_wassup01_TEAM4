
from typing import Literal
import numpy as np
import torch
import pandas as pd

def df2d_to_array3d(data):
  feature_size=data.iloc[:,2:].shape[1]
  time_size=len(data['date_time'].value_counts())
  sample_size=len(data.num.value_counts())
  return data.iloc[:,2:].values.reshape([sample_size, time_size, feature_size])

def preprocess(data, num_idx, tst_size, window_size, select_channel_idx, split:Literal['train', 'test']='train'):
  data=df2d_to_array3d(data) # (60, 2040, 8)
  data_df = pd.DataFrame(data[num_idx,:,:]) #(2040, 8)
  data_df['rolling_mean'] = data_df.iloc[:, 0].rolling(24).mean()
  data_df['diff'] = data_df.iloc[:, 0].diff(24)
  data_df = data_df.dropna() # (2017, 10)
  data = data_df.values

  if split == 'train': # trn, val
    train = data[:-tst_size-tst_size,:]
    test = data[-tst_size-tst_size-window_size:-tst_size,:]
  elif split == 'test': # trn, tst
    train = data[:-tst_size,:]
    test = data[-tst_size-window_size:,:]

  train = pd.DataFrame(train).iloc[:, select_channel_idx].to_numpy(dtype=np.float32)
  test = pd.DataFrame(test).iloc[:, select_channel_idx].to_numpy(dtype=np.float32)

  return train, test