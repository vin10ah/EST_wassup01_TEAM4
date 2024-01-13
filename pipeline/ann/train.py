import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics
from typing import Optional
import numpy as np

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
    X, y = X.flatten(1).to(device), y[:,:,0].to(device)
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item() * len(y)
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
    X, y = X.flatten(1).to(device), y[:,:,0].to(device)
    output = model(X)
    loss = criterion(output, y)
    if metrics is not None:
      metrics.update(output, y)

  return loss.item()

def evaluate_dynamic(
  model:nn.Module,
  criterion:callable,
  data_loader:DataLoader,
  tst_scale:np.array,
  tst_size=int,
  prediction_size=int,
  window_size=int,
  device:str='cpu',
)->float:
  '''evaluate

  Args:
    model: model
    criterions: list of criterion functions
    data_loader: data loader
    device: device
    metrcis: metrics
  '''
  prds = []
  X = torch.tensor(data_loader.dataset[0][0]).flatten().unsqueeze(dim=0).to(device)
  Y = torch.tensor(tst_scale[window_size:])
  
  model.eval()
  for _ in range(int(tst_size/prediction_size)):
    with torch.inference_mode():
      output = model(X).to(device)
      output = output.flatten()
    X = torch.concat([X.flatten(), output])[-window_size:]
    X = X.unsqueeze(dim=0)
    output = output.to('cpu')
    prds.append(output)
  prds = torch.tensor(np.concatenate(prds))
  loss = criterion(prds, Y.squeeze(dim=1))

  return loss.item()

def main(cfg):
  import pandas as pd
  from custom_ds import TimeseriesDataset
  from tqdm.auto import trange
  import matplotlib.pyplot as plt
  from collections import defaultdict
  
  from metric import mape, mae, R2_score, rmse
  from preprocess import preprocess


  train_params = cfg.get('train_params')
  device = torch.device(train_params.get('device'))
  
  # read_csv
  files = cfg.get('files')
  data = pd.read_csv(files.get('data'))

  # dataset
  dataset_params = train_params.get('dataset_params')
  window_size = dataset_params.get('window_size')
  prediction_size = dataset_params.get('prediction_size')
  
  # preprocess prams
  preprocess_params = cfg.get('preprocess_params')
  num_idx = preprocess_params.get('num_idx') # building num(one of 60)
  split = preprocess_params.get('split')
  tst_size = preprocess_params.get('tst_size')
  select_channel_idx = preprocess_params.get('select_channel_idx')
  c_n = len(select_channel_idx)
  scaler = preprocess_params.get('scaler')
  scaler2 = preprocess_params.get('scaler2')
  
  trn, tst = preprocess(data, num_idx, tst_size, window_size, select_channel_idx, split)
  
  # data scale
  scaler = scaler
  trn_scale = scaler.fit_transform(trn[:, :1])
  tst_scale = scaler.transform(tst[:, :1])
  
  if c_n >= 2:
    scaler2 = scaler2
    trn_m = scaler2.fit_transform(trn[:, 1:])
    trn_scale = np.concatenate((trn_scale, trn_m), axis=1)

    tst_m = scaler2.transform(tst[:, 1:])
    tst_scale = np.concatenate((tst_scale, tst_m), axis=1)
  
  # trn(dataset, dataloader)
  trn_dl_params = train_params.get('trn_data_loader_params')
  trn_ds = TimeseriesDataset(trn_scale, window_size, prediction_size)
  trn_dl = DataLoader(trn_ds, **trn_dl_params)

  # tst(dataset, dataloader)
  tst_dl_params = train_params.get('tst_data_loader_params')
  tst_ds = TimeseriesDataset(tst_scale, window_size, prediction_size)
  tst_dl_params['batch_size'] = len(tst_ds)
  tst_dl = DataLoader(tst_ds, **tst_dl_params)

  # setting model
  model = cfg.get('model')
  model_params = cfg.get('model_params')
  model_params['input_dim'] = window_size
  model_params['output_dim'] = prediction_size
  model_params['c_n'] = c_n
  model = model(**model_params).to(device)

  # lr_scheduler setting
  lr_scheduler = train_params.get('lr_scheduler')
  scheduler_params = train_params.get('scheduler_params')

  # optimizer
  Optim = train_params.get('optim')
  optim_params = train_params.get('optim_params')
  optimizer = Optim(model.parameters(), **optim_params)

  scheduler = lr_scheduler(optimizer, **scheduler_params)

  # loss_fn
  loss_fn = train_params.get('loss_fn')
  pbar = trange(train_params.get('epochs'))
  history = defaultdict(list)

  # predict_mode
  predict_mode = cfg.get('predict_mode')
  for _ in pbar:
    trn_loss = train_one_epoch(model, loss_fn, optimizer, trn_dl, device)
    
    if predict_mode == 'one_step':
      tst_loss = evaluate(model, loss_fn, tst_dl, device)
    elif predict_mode == 'dynamic':
      tst_loss = evaluate_dynamic(model, loss_fn, tst_dl, tst_scale, tst_size, prediction_size,window_size , device)

    # lr_scheduler
    #scheduler.step(tst_loss)
    
    history['trn_loss'].append(trn_loss)
    history['tst_loss'].append(tst_loss)
    pbar.set_postfix(trn_loss=trn_loss, tst_loss=tst_loss)

  if predict_mode == 'one_step':
    # eval
    model.eval()
    with torch.inference_mode():
      x, y = next(iter(tst_dl))
      x, y = x.flatten(1).to(device), y[:,:,0].to(device)
      prd = model(x)
    
    # inverse scale
    y = scaler.inverse_transform(y.cpu())
    prd = scaler.inverse_transform(prd.cpu())

    y = np.concatenate([y[:,0], y[-1,1:]])
    p = np.concatenate([prd[:,0], prd[-1,1:]])

  elif predict_mode == 'dynamic':
    prds = []
    X = torch.tensor(tst_dl.dataset[0][0]).flatten().unsqueeze(dim=0).to(device)
    Y = torch.tensor(tst_scale[window_size:])

    model.eval()
    for _ in range(int(tst_size/prediction_size)):
      with torch.inference_mode():
        output = model(X).to(device)
        output = output.flatten()
      X = torch.concat([X.flatten(), output])[-window_size:]
      X = X.unsqueeze(dim=0)
      output = output.to('cpu')
      prds.append(output)

    prds = torch.tensor(np.concatenate(prds))
    # inverse scale
    y = scaler.inverse_transform(Y)
    p = scaler.inverse_transform(prds.unsqueeze(dim=1))


  ###########
  ### log ###
  ###########
  output = cfg.get('output')
  log = output.get('output_log')

  # loss(train, test)
  tst_min = min(history['tst_loss'])
  min_idx = history['tst_loss'].index(tst_min)

  y1 = history['trn_loss']
  y2 = history['tst_loss']
  plt.figure(figsize=(8, 6))
  plt.plot(y1, color='#16344E', label='trn_loss')
  plt.plot(y2, color='#71706C', label='tst_loss')
  plt.legend()
  plt.title(f"Neural Network, Min_loss(test):{tst_min:.4f}, Min_idx(test):{min_idx}")
  plt.savefig(f'losses_{log}.png')

  # predict and metric
  plt.figure(figsize=(8, 6))
  plt.plot(range(tst_size), p, color='#16344E', label="Prediction")
  plt.plot(range(tst_size), y, color='#71706C', label="True")
  plt.legend()
  plt.title(f"Neural Network, MAPE:{mape(p,y):.4f}, MAE:{mae(p,y):.4f}, R2_SCORE:{R2_score(p,y):.4f}, RMSE:{rmse(p,y):.4f}")
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

'''