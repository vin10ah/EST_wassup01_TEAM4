from torch.utils.data import Dataset

class TimeseriesDataset(Dataset):
  def __init__(self, data, lookback_size=10, forecast_size=5):
    self.data = data
    self.lookback_size = lookback_size
    self.forecast_size = forecast_size

  def __len__(self):
    return len(self.data) - self.lookback_size - self.forecast_size + 1

  def __getitem__(self, i):
    idx = (i+self.lookback_size)
    x = self.data[i:idx]
    y = self.data[idx:idx+self.forecast_size]
    return x, y