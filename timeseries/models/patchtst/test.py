def test(cfg):
  import torch
  import pandas as pd
  import numpy as np

   # get train params
  train_params = cfg.get('train_params')

  # setting device
  device = torch.device(train_params.get('device'))

  # get test data
  files = cfg.get('files')
  X_tst = torch.tensor(pd.read_csv(files.get('X_tst')).to_numpy(dtype=np.float32), device=device)
  y_tst = torch.tensor(pd.read_csv(files.get('y_tst')).to_numpy(dtype=np.float32)).reshape(-1, 1)
  
  # setting model
  model = cfg.get('model')
  model_params = cfg.get('model_params')
  model_params['input_dim'] = X_tst.shape[-1]
  model = model(**model_params).to(device)

  trained_model = files.get('output')
  model.load_state_dict(torch.load(trained_model))

  # predict
  model.eval()
  with torch.inference_mode:
    pred = model(X_tst).cpu()
    pred = (pred > 0.5).float().flatten()
  
  # metrics