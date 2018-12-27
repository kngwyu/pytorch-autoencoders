from torch.utils.data import DataLoader, Dataset
from .base import AutoEncoderBase
from .config import Config
from typing import List


def train(ae: AutoEncoderBase, config: Config, data_set: Dataset) -> List[float]:
    data_loader = DataLoader(data_set, batch_size=config.batch_size, shuffle=True)
    optimizer = config.optim(ae.parameters())
    loss_list = []
    for epoch in range(config.num_epochs):
        for data in data_loader:
            img, _ = data
            img = img.to(config.device)
            res = ae(img)
            loss = config.criterion(res, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(float(loss.item()))
    return loss_list
