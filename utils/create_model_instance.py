from typing import Dict

from torch import nn

from models.ft_transformer import FTTransformer
from models.mlp import MLP
from models.resnet import ResNet


def create_model_instance(config: Dict) -> nn.Module:
    models = [MLP, FTTransformer, ResNet]

    for m in models:
        if config['model_name'] == m.__name__:
            return m(**config['model'])

    raise ValueError(f"Unsupported model_name \"{config['model_name']}\"")
