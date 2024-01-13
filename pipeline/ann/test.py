import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metric import mape, mae, R2_score, rmse
from preprocess import preprocess
from custom_ds import TimeseriesDataset

def test(cfg):
  # get data
  files = cfg.get('files')
  data = pd.read_csv(files.get('data'))

  output = cfg.get('output')
  

  #dataset
  train_params = cfg.get('train_params')
  dataset_params = train_params.get('dataset_params')
  window_size = dataset_params.get('window_size')
  prediction_size = dataset_params.get('prediction_size')

  device = torch.device(train_params.get('device'))

  # preprocess prams
  preprocess_params = cfg.get('preprocess_params')
  num_idx = preprocess_params.get('num_idx') # building num(one of 60)
  split = preprocess_params.get('split')
  tst_size = preprocess_params.get('tst_size')
  select_channel_idx = preprocess_params.get('select_channel_idx')
  c_n = len(select_channel_idx)
  

  _, tst = preprocess(data, num_idx, tst_size, window_size, select_channel_idx, split)

  # tst(dataset, dataloader)
  tst_dl_params = train_params.get('tst_data_loader_params')
  tst_ds = TimeseriesDataset(tst, window_size, prediction_size)
  tst_dl_params['batch_size'] = len(tst_ds)
  tst_dl = DataLoader(tst_ds, **tst_dl_params)

  # setting model
  model = cfg.get('model')
  model_params = cfg.get('model_params')
  model_params['input_dim'] = window_size
  model_params['output_dim'] = prediction_size
  model_params['c_n'] = c_n
  model = model(**model_params).to(device)

  model.load_state_dict(torch.load(output.get('load_pth_24')))

  # predict_mode
  predict_mode = cfg.get('predict_mode')

  if predict_mode == 'one_step':
    # eval
    model.eval()
    with torch.inference_mode():
      x, y = next(iter(tst_dl))
      x, y = x.flatten(1).to(device), y[:,:,0].to(device)
      prd = model(x)

    y, prd = y.cpu(), prd.cpu()
    y = np.concatenate([y[:,0], y[-1,1:]])
    p = np.concatenate([prd[:,0], prd[-1,1:]])

  elif predict_mode == 'dynamic':
    prds = []
    X = torch.tensor(tst_dl.dataset[0][0]).flatten().unsqueeze(dim=0).to(device)
    y = torch.tensor(tst[window_size:])

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


  ###########
  ### log ###
  ###########
  
  log = output.get('output_log')

  # predict and metric
  plt.figure(figsize=(8, 6))
  plt.plot(range(tst_size), p, color='#16344E', label="Prediction")
  plt.plot(range(tst_size), y, color='#71706C', label="True")
  plt.legend()
  plt.title(f"Neural Network, MAPE:{mape(p,y):.4f}, MAE:{mae(p,y):.4f}, R2_SCORE:{R2_score(p,y):.4f}, MSE:{mse(p,y):.4f}, RMSE:{rmse(p,y):.4f}")
  plt.savefig(f'predict_{log}.png')


def get_args_parser(add_help=True):
  import argparse
  
  parser = argparse.ArgumentParser(description="Pytorch test", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  exec(open(args.config).read())
  test(config)