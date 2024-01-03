import torch
from torch import nn
from torch.utils.data import DataLoader

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
  from custom_ds import TimeseriesDataset
  from tqdm.auto import trange
  import matplotlib.pyplot as plt
  import seaborn as sns
  from collections import defaultdict
  from sklearn.preprocessing import MinMaxScaler
  import statsmodels.api as sm # 임시 데이터

  from eval import evaluate
  from metric import mape, mae


  train_params = cfg.get('train_params')
  device = torch.device(train_params.get('device'))
  
  files = cfg.get('files')

  # dataset
  dataset_params = train_params.get('dataset_params')
  window_size = dataset_params.get('window_size')
  prediction_size = dataset_params.get('prediction_size')

  # 임시 데이터
  data = sm.datasets.sunspots.load_pandas().data
  data.index = pd.Index(sm.tsa.datetools.dates_from_range("1700", "2008"))
  data.index.freq = data.index.inferred_freq
  del data["YEAR"]

  tst_size = 20

  # data scale
  scaler = MinMaxScaler()
  trn_scaled = scaler.fit_transform(data[:-tst_size].to_numpy(dtype=np.float32)).flatten()
  tst_scaled = scaler.transform(data[-tst_size-window_size:].to_numpy(dtype=np.float32)).flatten()
  
  # trn(dataset, dataloader)
  trn_dl_params = train_params.get('trn_data_loader_params')
  trn_ds = TimeseriesDataset(trn_scaled, window_size, prediction_size)
  trn_dl = DataLoader(trn_ds, **trn_dl_params)

  # tst(dataset, dataloader)
  tst_dl_params = train_params.get('trn_data_loader_params')
  tst_ds = TimeseriesDataset(tst_scaled, window_size, prediction_size)
  tst_dl_params['batch_size'] = len(tst_ds)
  tst_dl = DataLoader(tst_ds, **tst_dl_params)

  # setting model
  model = cfg.get('model')
  model_params = cfg.get('model_params')
  model_params['input_dim'] = window_size
  model_params['output_dim'] = prediction_size
  model = model(**model_params).to(device)

  # lr_scheduler setting
  lr_scheduler = train_params.get('lr_scheduler')
  scheduler_params = train_params.get('scheduler_params')

  # optimizer
  Optim = train_params.get('optim')
  optim_params = train_params.get('optim_params')
  optimizer = Optim(model.parameters(), **optim_params)

  #scheduler = self.lr_scheduler(optim, **self.scheduler_kwargs)

  # loss_fn
  loss_fn = train_params.get('loss_fn')
  pbar = trange(train_params.get('epochs'))
  history = defaultdict(list)

  for _ in pbar:
    trn_loss = train_one_epoch(model, loss_fn, optimizer, trn_dl, device)
    tst_loss = evaluate(model, loss_fn, tst_dl, device)

    # lr_scheduler
    #scheduler.step(val_loss)
    
    history['trn_loss'].append(trn_loss)
    history['tst_loss'].append(tst_loss)
    pbar.set_postfix(trn_loss=trn_loss, tst_loss=tst_loss)

  # eval
  model.eval()
  with torch.inference_mode():
    x, y = next(iter(tst_dl))
    x, y = x.to(device), y.to(device)
    prd = model(x)
  
  # inverse scale
  y = scaler.inverse_transform(y.cpu())
  prd = scaler.inverse_transform(prd.cpu())

  y = np.concatenate([y[:,0], y[-1,1:]])
  p = np.concatenate([prd[:,0], prd[-1,1:]])

  ###########
  ### log ###
  ###########
  log = files.get('output_log')

  # loss(train, test)
  y1 = history['trn_loss']
  y2 = history['tst_loss']
  plt.figure(figsize=(8, 6))
  plt.plot(y1, color='#16344E', label='trn_loss')
  plt.plot(y2, color='#71706C', label='tst_loss')
  plt.legend()
  plt.title('losses')
  plt.savefig(f'losses_{log}.png')

  # predict and metric
  plt.figure(figsize=(8, 6))
  plt.plot(range(tst_size), p, color='#16344E', label="Prediction")
  plt.plot(range(tst_size), y, color='#71706C', label="True")
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

'''