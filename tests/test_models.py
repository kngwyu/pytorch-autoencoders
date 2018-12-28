import pytest
from pytorch_autoencoders import AutoEncoderBase, Config
from pytorch_autoencoders.models import AutoEncoder, VariationalAutoEncoder, VaeOutPut
import torch

BATCH_SIZE = 10


@pytest.mark.parametrize('aegen', [AutoEncoder, VariationalAutoEncoder])
def test_model_dim(aegen: AutoEncoderBase) -> None:
    input_dim = torch.Size((28, 28))
    config = Config()
    ae = aegen(input_dim, config)
    batch = torch.randn(BATCH_SIZE, *input_dim)
    with torch.no_grad():
        res = ae(batch.to(config.device))
    if isinstance(res, VaeOutPut):
        res = res.x
    assert res.shape == torch.Size((BATCH_SIZE, *input_dim))
