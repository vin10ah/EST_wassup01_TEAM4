
from typing import Literal
import numpy as np
import torch
import pandas as pd

def df2d_to_array3d(data):
  feature_size=data.iloc[:,2:].shape[1]
  time_size=len(data['date_time'].value_counts())
  sample_size=len(data.num.value_counts())
  return data.iloc[:,2:].values.reshape([sample_size, time_size, feature_size])

def preprocess(data, num_idx, tst_size, window_size, scaler_cfg, patch_length, n_patches, prediction_length,split:Literal['train', 'test']='train'):
  from patchtsdataset import PatchTSDataset

  data=df2d_to_array3d(data) # (60, 2040, 8)
  data_df = pd.DataFrame(data[num_idx,:,:])
  data_df['rolling_mean'] = data_df.iloc[:, 0].rolling(24).mean()
  data_df['diff'] = data_df.iloc[:, 0].diff(24)
  data_df = data_df.dropna() # (2017, 9)
  data = torch.tensor(data_df.values)
 
  if split == 'train': # trn, val
    train = data[:-tst_size-tst_size,:]
    test = data[-tst_size-tst_size-window_size:-tst_size,:]
    
  elif split == 'test': # trn, tst
    train = data[:-tst_size,:]
    test = data[-tst_size-window_size:,:]


  train_target = train[:,0].unsqueeze(1).numpy().astype(np.float32)
  test_target = test[:,0].unsqueeze(1).numpy().astype(np.float32)

  channel1 = train[:,1].unsqueeze(1).numpy().astype(np.float32)
  channel2 = train[:,2].unsqueeze(1).numpy().astype(np.float32)
  channel3 = train[:,3].unsqueeze(1).numpy().astype(np.float32)
  channel4 = train[:,4].unsqueeze(1).numpy().astype(np.float32)
  channel5 = train[:,5].unsqueeze(1).numpy().astype(np.float32)
  channel6 = train[:,6].unsqueeze(1).numpy().astype(np.float32)
  channel7 = train[:,7].unsqueeze(1).numpy().astype(np.float32)
  channel8 = train[:,8].unsqueeze(1).numpy().astype(np.float32)

  scaler1 = scaler_cfg # temp
  channel1 = scaler1.fit_transform(channel1).flatten()

  scaler2 = scaler_cfg # wind_speed
  channel2 = scaler2.fit_transform(channel2).flatten()

  scaler3 = scaler_cfg # humidity
  channel3 = scaler3.fit_transform(channel3).flatten()

  scaler4 = scaler_cfg # rainfall
  channel4 = scaler4.fit_transform(channel4).flatten()

  scaler5 = scaler_cfg # sunshine
  channel5 = scaler5.fit_transform(channel5).flatten()

  scaler6 = scaler_cfg # no_elec
  channel6 = scaler6.fit_transform(channel6).flatten()

  scaler7 = scaler_cfg # sunlight_have
  channel7 = scaler7.fit_transform(channel7).flatten()

  scaler8 = scaler_cfg # rolling_mean
  channel8 = scaler8.fit_transform(channel8).flatten()

  channel1_ds = PatchTSDataset(channel1, patch_length, n_patches, prediction_length)
  channel2_ds = PatchTSDataset(channel2, patch_length, n_patches, prediction_length)
  channel3_ds = PatchTSDataset(channel3, patch_length, n_patches, prediction_length)
  channel4_ds = PatchTSDataset(channel4, patch_length, n_patches, prediction_length)
  channel5_ds = PatchTSDataset(channel5, patch_length, n_patches, prediction_length)
  channel6_ds = PatchTSDataset(channel6, patch_length, n_patches, prediction_length)
  channel7_ds = PatchTSDataset(channel7, patch_length, n_patches, prediction_length)
  channel8_ds = PatchTSDataset(channel8, patch_length, n_patches, prediction_length)

  channel_ds_list = [channel1_ds, channel2_ds, channel3_ds, channel4_ds, channel5_ds, channel6_ds, channel7_ds, channel8_ds]

  return train_target, test_target, channel_ds_list