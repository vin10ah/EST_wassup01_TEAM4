def test(cfg):
  import torch
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  from torch.utils.data import DataLoader

  from preprocess import preprocess
  from patchtsdataset import PatchTSDataset
  from metric import mape, mae, R2_score, rmse

  train_params = cfg.get('train_params')
  device = torch.device(train_params.get('device'))
  output = cfg.get('output')

  model = cfg.get('model')
  model_params = cfg.get('model_params')

  patch_length = model_params.get('input_dim')
  n_patches = model_params.get('n_token')
  window_size = int(patch_length * n_patches / 2)
  prediction_length = model_params.get('output_dim')

  # get data
  files = cfg.get('files')
  data = pd.read_csv(files.get('data'))

  # preprocess prams
  preprocess_params = cfg.get('preprocess_params')
  num_idx = preprocess_params.get('num_idx') # building num(one of 60)
  split = preprocess_params.get('split')
  tst_size = preprocess_params.get('tst_size')

  # preprocess
  _, tst, _ = preprocess(data, num_idx, tst_size, window_size, preprocess_params.get('scaler'), patch_length, n_patches, prediction_length, split)
  tst = tst.flatten()
  tst_dl_params = train_params.get('tst_data_loader_params')
  tst_ds = PatchTSDataset(tst, patch_length, n_patches, prediction_length)
  tst_dl_params['batch_size'] = len(tst_ds)
  tst_dl = DataLoader(tst_ds, **tst_dl_params)

  model = model(**model_params).to(device)
  model.load_state_dict(torch.load(output.get('load_pth_24')))
  
  # eval
  model.eval()
  with torch.inference_mode():
    x, y = next(iter(tst_dl))
    x, y = x.to(device), y.to(device)
    p = model(x)
  y, p = y.cpu(), p.cpu()
  
  y = np.concatenate([y[:,0], y[-1,1:]])
  p = np.concatenate([p[:,0], p[-1,1:]])

  # log
  log = output.get('output_log')

  # predict and metric
  predict_range = len(y)
  plt.figure(figsize=(8, 6))
  plt.plot(range(predict_range), y, label="True")
  plt.plot(range(predict_range), p, label="Prediction")
  plt.legend()
  plt.title(f"PatchTST, MAPE:{mape(p,y):.4f}, MAE:{mae(p,y):.4f}, R2_SCORE:{R2_score(p,y):.4f}, RMSE:{rmse(p,y):.4f}")
  plt.savefig(f'predict_{log}.png')

def get_args_parser(add_help=True):
  import argparse

  parser = argparse.ArgumentParser(description="Pytorch train", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")

  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  exec(open(args.config).read())
  test(config)