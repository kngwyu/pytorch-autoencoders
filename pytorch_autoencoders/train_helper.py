import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Any, Callable
from .base import AutoEncoderBase
from .config import Config
from .models import VaeOutPut


def simple_logfn(loss: torch.Tensor, *args) -> dict:
    return dict(loss_mean=loss.detach().cpu().numpy().item())


def vae_logfn(loss: torch.Tensor, out: VaeOutPut) -> dict:
    logvar = out.logvar.detach()
    return dict(
        loss_mean=loss.detach().item(),
        mu_mean=out.logvar.detach().mean().item(),
        logvar_mean=out.logvar.detach().mean().item(),
        var_mean=logvar.exp().mean().item(),
    )


def _onehot(num_clases: int, device: torch.device) -> torch.Tensor:
    """Returns label to onehot converter
    """

    def _encode(labels: torch.Tensor):
        batch_size = labels.size(0)
        indices = torch.arange(batch_size).to(device)
        y = torch.zeros(batch_size, num_clases).to(device)
        y[indices, labels] = 1.0
        return y

    return _encode


def train(
    ae: AutoEncoderBase,
    config: Config,
    data_set: Dataset,
    log_fn: Callable[[torch.Tensor, Any], dict] = simple_logfn,
) -> pd.DataFrame:
    data_loader = DataLoader(data_set, batch_size=config.batch_size, shuffle=True)
    optimizer = config.optim(ae.parameters())
    df = pd.DataFrame()
    print("Started training...")
    for epoch in range(config.num_epochs):
        epoch_df = pd.DataFrame()
        for i, data in enumerate(data_loader):
            img, _ = data
            img = img.to(config.device)
            res = ae(img)
            loss = config.criterion(res, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_df = epoch_df.append(
                pd.Series(log_fn(loss, res), name="{}:{}".format(i, epoch))
            )
        print("epoch: ", epoch)
        print(epoch_df.mean())
        df = df.append(epoch_df)
        if hasattr(config.criterion, "update"):
            config.criterion.update()
    return df


def train_ss(
    ae: AutoEncoderBase,
    config: Config,
    data_set: Dataset,
    log_fn: Callable[[torch.Tensor, Any], dict] = simple_logfn,
    labels_per_class: int = 100,
) -> pd.DataFrame:
    data_loader = DataLoader(data_set, batch_size=config.batch_size, shuffle=True)
    all_data = len(data_loader)
    supervised_indices = np.random.choice(all_data, labels_per_class)
    supervised_next = 0
    alpha = config.alpha_coef * (all_data - labels_per_class) / labels_per_class
    optimizer = config.optim(ae.parameters())
    df = pd.DataFrame()
    print("Started training...")
    onehot = _onehot(10, config.device)
    for epoch in range(config.num_epochs):
        epoch_df = pd.DataFrame()
        for i, data in enumerate(data_loader):
            if i == supervised_indices[supervised_next]:
                img, label = data
                label = onehot(label)
                has_label = True
            else:
                img, _ = data
                label = None
                has_label = False
            img = img.to(config.device)
            res, img, label = ae(img, label)
            loss = config.criterion(res, img, label, alpha, not has_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_df = epoch_df.append(
                pd.Series(log_fn(loss, res), name="{}:{}".format(i, epoch))
            )
        print("epoch: ", epoch)
        print(epoch_df.mean())
        df = df.append(epoch_df)
        if hasattr(config.criterion, "update"):
            config.criterion.update()
    return df


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
    print("test_loss: {}".format(loss))
    return loss


def test_loss_ss(ae: AutoEncoderBase, config: Config, data_set: Dataset) -> float:
    data_loader = DataLoader(data_set, batch_size=config.batch_size, shuffle=True)
    cnt = 0
    epoch_loss = 0.0
    onehot = _onehot(10, config.device)
    for data in data_loader:
        img, label = data
        label = onehot(label)
        img = img.to(config.device)
        with torch.no_grad():
            res, _, _ = ae(img, label)
        loss = config.criterion(res, img, label)
        epoch_loss += float(loss.item())
        cnt += 1
    loss = epoch_loss / float(cnt)
    print("test_loss: {}".format(loss))
    return loss
