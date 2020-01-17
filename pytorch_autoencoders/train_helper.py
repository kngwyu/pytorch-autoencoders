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
