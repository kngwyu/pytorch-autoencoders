from torch.utils.data import DataLoader, Dataset
from .base import AutoEncoderBase
from .config import Config
from tqdm import tqdm
from typing import List


def train(ae: AutoEncoderBase, config: Config, data_set: Dataset) -> List[float]:
    data_loader = DataLoader(data_set, batch_size=config.batch_size, shuffle=True)
    optimizer = config.optim(ae.parameters())
    loss_list = []
    print('Started training...')
    for epoch in tqdm(range(config.num_epochs)):
        epoch_loss = 0.0
        for data in data_loader:
            img, _ = data
            img = img.to(config.device)
            print(img)
            res = ae(img)
            loss = config.criterion(res, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        loss = epoch_loss / float(config.batch_size)
        loss_list.append(loss)
    return loss_list
