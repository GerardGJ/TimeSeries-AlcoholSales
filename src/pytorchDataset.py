import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data, backcast_length, forecast_length):
        self.data = data
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

    def __len__(self):
        return len(self.data) - self.backcast_length - self.forecast_length

    def __getitem__(self, index):
        x = self.data[index : index + self.backcast_length]
        y = self.data[index + self.backcast_length : index + self.backcast_length + self.forecast_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
