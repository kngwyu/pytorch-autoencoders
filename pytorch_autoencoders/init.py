from functools import partial
from torch import nn, Tensor
from typing import Callable


def orthogonal(nonlinearity: str = 'relu') -> Callable[[nn.Module], None]:
    gain = nn.init.calculate_gain(nonlinearity)
    return partial(_init, weight_init=partial(nn.init.orthogonal_, gain=gain))


def kaiming_normal(nonlinearity: str = 'relu') -> Callable[[nn.Module], None]:
    return partial(_init, weight_init=partial(nn.init.kaiming_normal_, nonlinearity=nonlinearity))


def kaiming_uniform(nonlinearity: str = 'relu') -> Callable[[nn.Module], None]:
    return partial(_init, weight_init=partial(nn.init.kaiming_uniform_, nonlinearity=nonlinearity))


def _init(
        mod: nn.Module,
        weight_init: Callable[[Tensor], None],
        bias_init: Callable[[Tensor], None] = partial(nn.init.constant_, val = 0.0),
) -> None:
    for name, param in mod.named_parameters():
        if 'weight' in name:
            weight_init(param)
        if 'bias' in name:
            bias_init(param)
