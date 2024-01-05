import torch
from torch import nn
from torch.utils.data import DataLoader
from collections import defaultdict

def train_one_epoch(
  model:nn.Module,
  criterion:callable,
  optimizer:torch.optim.Optimizer,
  data_loader:DataLoader,
  device:str
) -> float:
  '''train one epoch
  
  Args:
      model: model
      criterion: loss
      optimizer: optimizer
      data_loader: data loader
      device: device
  '''
  model.train()
  total_loss = 0.
  for X, y in data_loader:
    X, y = X.to(device), y.to(device)
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item() * len(X)
  return total_loss/len(data_loader.dataset)

def main(cfg):
  import numpy as np
  import pandas as pd
  from tqdm.auto import trange
  import matplotlib.pyplot as plt
  import seaborn as sns

  from patchtsdataset import PatchTSDataset
  from eval import evaluate
  from sklearn.preprocessing import MinMaxScaler
  from metric import mape, mae

  
  train_params = cfg.get('train_params')
  device = torch.device(train_params.get('device'))

  model = cfg.get('model')
  model_params = cfg.get('model_params')

  patch_length = model_params.get('input_dim')
  n_patches = model_params.get('n_token')
  window_size = int(patch_length * n_patches / 2)
  prediction_length = model_params.get('output_dim')
  
  # read_csv
  files = cfg.get('files')
  trn = pd.read_csv('../../data/train.csv')
  
  def df2d_to_array3d(df_2d):
    feature_size=df_2d.iloc[:,2:].shape[1]
    time_size=len(df_2d['date_time'].value_counts())
    sample_size=len(df_2d.num.value_counts())
    return df_2d.iloc[:,2:].values.reshape([sample_size, time_size, feature_size])
  
  train=torch.tensor(df2d_to_array3d(trn))
  tst_size = int(2040 * .1)
  trn_0, tst_0 = train[0,:-tst_size,0].unsqueeze(1).numpy().astype(np.float32), train[0,-tst_size-window_size:,0].unsqueeze(1).numpy().astype(np.float32)
  trn_1, tst_1 = train[0,:-tst_size,1].unsqueeze(1).numpy().astype(np.float32), train[0,-tst_size-window_size:,1].unsqueeze(1).numpy().astype(np.float32)
  trn_2, tst_2 = train[0,:-tst_size,2].unsqueeze(1).numpy().astype(np.float32), train[0,-tst_size-window_size:,2].unsqueeze(1).numpy().astype(np.float32)
  trn_3, tst_3 = train[0,:-tst_size,3].unsqueeze(1).numpy().astype(np.float32), train[0,-tst_size-window_size:,3].unsqueeze(1).numpy().astype(np.float32)
  trn_4, tst_4 = train[0,:-tst_size,4].unsqueeze(1).numpy().astype(np.float32), train[0,-tst_size-window_size:,4].unsqueeze(1).numpy().astype(np.float32)
  trn_5, tst_5 = train[0,:-tst_size,5].unsqueeze(1).numpy().astype(np.float32), train[0,-tst_size-window_size:,5].unsqueeze(1).numpy().astype(np.float32)
  trn_6, tst_6 = train[0,:-tst_size,6].unsqueeze(1).numpy().astype(np.float32), train[0,-tst_size-window_size:,6].unsqueeze(1).numpy().astype(np.float32)

  
  scaler0 = MinMaxScaler()
  trn0 = scaler0.fit_transform(trn_0).flatten()
  tst0 = scaler0.transform(tst_0).flatten()

  scaler1 = MinMaxScaler()
  trn1 = scaler1.fit_transform(trn_1).flatten()
  tst1 = scaler1.transform(tst_1).flatten()

  scaler2 = MinMaxScaler()
  trn2 = scaler2.fit_transform(trn_2).flatten()
  tst2 = scaler2.transform(tst_2).flatten()

  scaler3 = MinMaxScaler()
  trn3 = scaler3.fit_transform(trn_3).flatten()
  tst3 = scaler3.transform(tst_3).flatten()

  scaler4 = MinMaxScaler()
  trn4 = scaler4.fit_transform(trn_4).flatten()
  tst4 = scaler4.transform(tst_4).flatten()

  scaler5 = MinMaxScaler()
  trn5 = scaler5.fit_transform(trn_5).flatten()
  tst5 = scaler5.transform(tst_5).flatten()

  scaler6 = MinMaxScaler()
  trn6 = scaler6.fit_transform(trn_6).flatten()
  tst6 = scaler6.transform(tst_6).flatten()

  trn_dl_params = train_params.get('trn_data_loader_params')
  trn_ds0 = PatchTSDataset(trn0, patch_length, n_patches, prediction_length)
  trn_ds1 = PatchTSDataset(trn1, patch_length, n_patches, prediction_length)
  trn_ds2 = PatchTSDataset(trn2, patch_length, n_patches, prediction_length)
  trn_ds3 = PatchTSDataset(trn3, patch_length, n_patches, prediction_length)
  trn_ds4 = PatchTSDataset(trn4, patch_length, n_patches, prediction_length)
  trn_ds5 = PatchTSDataset(trn5, patch_length, n_patches, prediction_length)
  trn_ds6 = PatchTSDataset(trn6, patch_length, n_patches, prediction_length)

  trn_ds = torch.utils.data.ConcatDataset([trn_ds0, trn_ds1, trn_ds2, trn_ds3, trn_ds4, trn_ds5, trn_ds6])
  trn_dl = DataLoader(trn_ds, **trn_dl_params)

  tst_dl_params = train_params.get('tst_data_loader_params')
  tst_ds0 = PatchTSDataset(tst0, patch_length, n_patches, prediction_length)
  tst_ds1 = PatchTSDataset(tst1, patch_length, n_patches, prediction_length)
  tst_ds2 = PatchTSDataset(tst2, patch_length, n_patches, prediction_length)
  tst_ds3 = PatchTSDataset(tst3, patch_length, n_patches, prediction_length)
  tst_ds4 = PatchTSDataset(tst4, patch_length, n_patches, prediction_length)
  tst_ds5 = PatchTSDataset(tst5, patch_length, n_patches, prediction_length)
  tst_ds6 = PatchTSDataset(tst6, patch_length, n_patches, prediction_length)

  tst_ds = torch.utils.data.ConcatDataset([tst_ds0,tst_ds1,tst_ds2,tst_ds3,tst_ds4,tst_ds5,tst_ds6])
  tst_dl_params['batch_size'] = len(tst_ds)
  tst_dl = DataLoader(tst_ds, **tst_dl_params)

  # # trn(dataset, dataloader)
  # trn_dl_params = train_params.get('trn_data_loader_params')
  # trn_ds = PatchTSDataset(trn1, patch_length, n_patches)
  # trn_dl = DataLoader(trn_ds, **trn_dl_params)

  # # tst(dataset, dataloader)
  # tst_dl_params = train_params.get('tst_data_loader_params')
  # tst_ds = PatchTSDataset(tst1, patch_length, n_patches)
  # tst_dl_params['batch_size'] = len(tst_ds)
  # tst_dl = DataLoader(tst_ds, **tst_dl_params)

  model = model(**model_params).to(device)

  Optim = train_params.get('optim')
  optim_params = train_params.get('optim_params')
  optimizer = Optim(model.parameters(), **optim_params)

  loss_fn = train_params.get('loss_fn')
  pbar = trange(train_params.get('epochs'))
  history = defaultdict(list)
  for _ in pbar:
    trn_loss = train_one_epoch(model, loss_fn, optimizer, trn_dl, device)
    tst_loss = evaluate(model, loss_fn, tst_dl, device)
    history['trn_loss'].append(trn_loss)
    history['tst_loss'].append(tst_loss)
    pbar.set_postfix(trn_loss=trn_loss, tst_loss=tst_loss)
  
  # eval
  model.eval()
  with torch.inference_mode():
    x, y = next(iter(tst_dl))
    x, y = x.to(device), y.to(device)
    p = model(x)

  y = scaler0.inverse_transform(y.cpu())
  p = scaler0.inverse_transform(p.cpu())
  
  y = np.concatenate([y[:,0], y[-1,1:]])
  p = np.concatenate([p[:,0], p[-1,1:]])
  
  # log
  log = files.get('output_log')

  # loss
  y1 = history['trn_loss']
  y2 = history['tst_loss']
  plt.figure(figsize=(8, 6))
  plt.plot(y1, color='#16344E', label='trn_loss')
  plt.plot(y2, color='#71706C', label='tst_loss')
  plt.legend()
  plt.title('losses')
  plt.savefig(f'losses_{log}.png')

  # predict and metric
  predict_range = len(y)
  plt.figure(figsize=(8, 6))
  plt.plot(range(predict_range), y, label="True")
  plt.plot(range(predict_range), p, label="Prediction")
  plt.legend()
  plt.title(f"Neural Network, MAPE:{mape(p,y):.4f}, MAE:{mae(p,y):.4f}")
  plt.savefig(f'predict_{log}.png')

  # model
  torch.save(model.state_dict(), f'./{log}_model.pth')

def get_args_parser(add_help=True):
  import argparse
  
  parser = argparse.ArgumentParser(description="Pytorch train", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  exec(open(args.config).read())
  main(config)


  #TODO
  '''
  장기예측

  '''