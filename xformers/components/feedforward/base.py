from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch.nn as nn


class Activations(str, Enum):
    GeLU = "gelu"
    ReLU = "relu"


@dataclass
class FeedforwardConfig:
    dim_latent: int
    dropout: float
    activation: Activations
    hidden_layer_multiplier: int


# Define the common interface, every feedforward block needs to derive from it
class Feedforward(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        dim_latent: Optional[int] = None,
        dropout: Optional[float] = None,
        activation: Optional[Activations] = None,
        hidden_layer_multiplier: Optional[int] = None,
    ):
        super().__init__()

    @classmethod
    @abstractmethod
    def from_config(cls, config: FeedforwardConfig) -> "Feedforward":
        # Could be that this handles the construction of the children, TBD
        pass
