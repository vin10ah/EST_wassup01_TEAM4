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
  

  from eval import evaluate
  from metric import mape, mae


  train_params = cfg.get('train_params')
  device = torch.device(train_params.get('device'))
  
  # read_csv
  files = cfg.get('files')
  trn = pd.read_csv(files.get('X_csv')) 

  # dataset
  dataset_params = train_params.get('dataset_params')
  window_size = dataset_params.get('window_size')
  prediction_size = dataset_params.get('prediction_size')

  # data split
  def df2d_to_array3d(df_2d):
    feature_size=df_2d.iloc[:,2:].shape[1] # 8
    time_size=len(df_2d['date_time'].value_counts()) # 2040
    sample_size=len(df_2d.num.value_counts()) # 60
    return df_2d.iloc[:,2:].values.reshape([sample_size, time_size, feature_size]) # (60, 2040, 8)
  
  train=torch.tensor(df2d_to_array3d(trn))
  tst_size = int(2040 * .1)

  # channel
  trn_0, tst_0 = train[0,:-tst_size,0].unsqueeze(1).numpy().astype(np.float32), train[0,-tst_size-window_size:,0].unsqueeze(1).numpy().astype(np.float32)
  # trn_1, tst_1 = train[0,:-tst_size,1].unsqueeze(1).numpy().astype(np.float32), train[0,-tst_size-window_size:,1].unsqueeze(1).numpy().astype(np.float32)
  # trn_2, tst_2 = train[0,:-tst_size,2].unsqueeze(1).numpy().astype(np.float32), train[0,-tst_size-window_size:,2].unsqueeze(1).numpy().astype(np.float32)
  # trn_3, tst_3 = train[0,:-tst_size,3].unsqueeze(1).numpy().astype(np.float32), train[0,-tst_size-window_size:,3].unsqueeze(1).numpy().astype(np.float32)
  # trn_4, tst_4 = train[0,:-tst_size,4].unsqueeze(1).numpy().astype(np.float32), train[0,-tst_size-window_size:,4].unsqueeze(1).numpy().astype(np.float32)
  # trn_5, tst_5 = train[0,:-tst_size,5].unsqueeze(1).numpy().astype(np.float32), train[0,-tst_size-window_size:,5].unsqueeze(1).numpy().astype(np.float32)
  # trn_6, tst_6 = train[0,:-tst_size,6].unsqueeze(1).numpy().astype(np.float32), train[0,-tst_size-window_size:,6].unsqueeze(1).numpy().astype(np.float32)

  # data scale
  scaler0 = MinMaxScaler()
  trn0 = scaler0.fit_transform(trn_0).flatten()
  tst0 = scaler0.transform(tst_0).flatten()
  
  # scaler1 = MinMaxScaler()
  # trn1 = scaler1.fit_transform(trn_1).flatten()
  # tst1 = scaler1.transform(tst_1).flatten()

  # scaler2 = MinMaxScaler()
  # trn2 = scaler2.fit_transform(trn_2).flatten()
  # tst2 = scaler2.transform(tst_2).flatten()

  # scaler3 = MinMaxScaler()
  # trn3 = scaler3.fit_transform(trn_3).flatten()
  # tst3 = scaler3.transform(tst_3).flatten()

  # scaler4 = MinMaxScaler()
  # trn4 = scaler4.fit_transform(trn_4).flatten()
  # tst4 = scaler4.transform(tst_4).flatten()

  # scaler5 = MinMaxScaler()
  # trn5 = scaler5.fit_transform(trn_5).flatten()
  # tst5 = scaler5.transform(tst_5).flatten()

  # scaler6 = MinMaxScaler()
  # trn6 = scaler6.fit_transform(trn_6).flatten()
  # tst6 = scaler6.transform(tst_6).flatten()
  
  # trn(dataset, dataloader)
  trn_dl_params = train_params.get('trn_data_loader_params')
  trn_ds = TimeseriesDataset(trn0, window_size, prediction_size)
  trn_dl = DataLoader(trn_ds, **trn_dl_params)

  # tst(dataset, dataloader)
  tst_dl_params = train_params.get('tst_data_loader_params')
  tst_ds = TimeseriesDataset(tst0, window_size, prediction_size)
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
  y = scaler0.inverse_transform(y.cpu())
  prd = scaler0.inverse_transform(prd.cpu())

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