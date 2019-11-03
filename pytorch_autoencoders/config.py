import torch
from .init import kaiming_normal


class Config:
    def __init__(self) -> None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.criterion = torch.nn.MSELoss()
        self.optim = torch.optim.Adam
        self.num_epochs = 100
        self.batch_size = 128
        self.initializer = kaiming_normal()
