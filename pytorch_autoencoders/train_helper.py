from torch.utils.data import DataLoader, Dataset
from .base import AutoEncoderBase
from .config import Config
import torch
from typing import List


def train(ae: AutoEncoderBase, config: Config, data_set: Dataset) -> List[float]:
    data_loader = DataLoader(data_set, batch_size=config.batch_size, shuffle=True)
    optimizer = config.optim(ae.parameters())
    loss_list = []
    print('Started training...')
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        cnt = 0
        for data in data_loader:
            img, _ = data
            img = img.to(config.device)
            res = ae(img)
            loss = config.criterion(res, img)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), config.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.item())
            cnt += 1
        loss = epoch_loss / float(cnt)
        print('epoch: {} loss: {}'.format(epoch, loss))
        loss_list.append(loss)
    return loss_list


def test_loss(ae: AutoEncoderBase, config: Config, data_set: Dataset) -> float:
    data_loader = DataLoader(data_set, batch_size=config.batch_size, shuffle=True)
    cnt = 0
    epoch_loss = 0.0
    for data in data_loader:
        img, _ = data
        img = img.to(config.device)
        with torch.no_grad():
            res = ae(img)
        loss = config.criterion(res, img)
        epoch_loss += float(loss.item())
        cnt += 1
    loss = epoch_loss / float(cnt)
    print('test_loss: {}'.format(loss))
    return loss
