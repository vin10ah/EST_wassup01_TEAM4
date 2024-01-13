import torch
from torch import nn
from torch.utils.data import DataLoader
from collections import defaultdict
import torchmetrics
from typing import Optional

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

def evaluate(
  model:nn.Module,
  criterion:callable,
  data_loader:DataLoader,
  device:str='cpu',
  metrics:Optional[torchmetrics.metric.Metric]=None
) -> float:
  '''evaluate
  
  Args:
      model: model
      criterions: list of criterion functions
      data_loader: data loader
      device: device
      metrcis: metrics
  '''
  model.eval()
  with torch.inference_mode():
      X, y = next(iter(data_loader))
      X, y = X.to(device), y.to(device)
      output = model(X)
      loss = criterion(output, y)
      if metrics is not None:
        metrics.update(output, y)

  return loss.item()

def main(cfg):
  import numpy as np
  import pandas as pd
  from tqdm.auto import trange
  import matplotlib.pyplot as plt

  from patchtsdataset import PatchTSDataset
  from metric import mape, mae, R2_score, rmse
  from preprocess import preprocess

  train_params = cfg.get('train_params')
  device = torch.device(train_params.get('device'))
  output = cfg.get('output')

  model = cfg.get('model')
  model_params = cfg.get('model_params')

  patch_length = model_params.get('input_dim')
  n_patches = model_params.get('n_token')
  window_size = int(patch_length * n_patches / 2)
  prediction_length = model_params.get('output_dim')
  
  # read_csv
  files = cfg.get('files')
  data = pd.read_csv(files.get('data'))

  # preprocess prams
  preprocess_params = cfg.get('preprocess_params')
  num_idx = preprocess_params.get('num_idx') # building num(one of 60)
  split = preprocess_params.get('split')
  tst_size = preprocess_params.get('tst_size')

  # preprocess
  trn, tst, channel_ds_list = preprocess(data, num_idx, tst_size, window_size, preprocess_params.get('scaler'), patch_length, n_patches, prediction_length, split)

  # elec_amount scale
  scaler = preprocess_params.get('scaler')
  trn = scaler.fit_transform(trn).flatten()
  tst = scaler.transform(tst).flatten()

  select_channel_idx = preprocess_params.get('select_channel_idx')

  # select channel
  channel_ds = [channel_ds_list[i] for i in select_channel_idx]
  trn_dl_params = train_params.get('trn_data_loader_params')
  trn_ds = PatchTSDataset(trn, patch_length, n_patches, prediction_length)
  trn_ds = torch.utils.data.ConcatDataset([trn_ds, *channel_ds])
  trn_dl = DataLoader(trn_ds, **trn_dl_params)

  tst_dl_params = train_params.get('tst_data_loader_params')
  tst_ds = PatchTSDataset(tst, patch_length, n_patches, prediction_length)
 
  tst_dl_params['batch_size'] = len(tst_ds)
  tst_dl = DataLoader(tst_ds, **tst_dl_params)

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

  y = scaler.inverse_transform(y.cpu())
  p = scaler.inverse_transform(p.cpu())
  
  y = np.concatenate([y[:,0], y[-1,1:]])
  p = np.concatenate([p[:,0], p[-1,1:]])
  
  # log
  log = output.get('output_log')

  # loss
  tst_min = min(history['tst_loss'])
  min_idx = history['tst_loss'].index(tst_min)

  y1 = history['trn_loss']
  y2 = history['tst_loss']
  plt.figure(figsize=(8, 6))
  plt.plot(y1, color='#16344E', label='trn_loss')
  plt.plot(y2, color='#71706C', label='tst_loss')
  plt.legend()
  plt.title(f"PatchTST Losses, Min_loss(test):{tst_min:.4f}, Min_idx(test):{min_idx}")
  plt.savefig(f'losses_{log}.png')

  # predict and metric
  predict_range = len(y)
  plt.figure(figsize=(8, 6))
  plt.plot(range(predict_range), y, label="True")
  plt.plot(range(predict_range), p, label="Prediction")
  plt.legend()
  plt.title(f"PatchTST, MAPE:{mape(p,y):.4f}, MAE:{mae(p,y):.4f}, R2_SCORE:{R2_score(p,y):.4f}, RMSE:{rmse(p,y):.4f}")
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
