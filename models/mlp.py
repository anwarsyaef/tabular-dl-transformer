import math
from collections import OrderedDict
from typing import List, Optional

import torch
from torch import nn


class MLPBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_prob: Optional[float] = None):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob is not None else None
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    def forward(self, x: torch.Tensor):
        x = self.relu(self.linear(x))

        if self.dropout is not None:
            x = self.dropout(x)

        return x


class SimpleTabularEmbedding(nn.Module):
    """Embedding layer to create an initial embedding for numerical and categorical features. The result is a flattened
    embedding containing the original numerical features, and the categorical features are appended using a lookup table
    of dimension embedding_dim."""

    def __init__(self, categories: List[int], embedding_dim):
        super().__init__()

        self.category_embeddings: Optional[nn.Embedding] = None
        if len(categories) > 0:
            self.register_buffer('category_offsets', torch.tensor([0] + categories[:-1]).cumsum(0))
            self.category_embeddings = nn.Embedding(sum(categories), embedding_dim)

        self.initialize_weights()

    def initialize_weights(self):
        if self.category_embeddings is not None:
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

    def forward(self, x_num: Optional[torch.FloatTensor], x_cat: Optional[torch.IntTensor]):
        assert x_num is not None or x_cat is not None
        x = []

        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x_cat = self.category_embeddings(x_cat + self.get_buffer('category_offsets'))
            # concatenate last two dimensions
            x_cat = x_cat.flatten(1, -1)
            x.append(x_cat)

        return torch.cat(x, dim=-1)


class MLP(nn.Module):
    def __init__(self, in_features_num: int, categories: List[int], block_out_features: list[int], embedding_dim: int,
                 out_features, dropout_prob: Optional[float] = None):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        in_features = in_features_num + len(categories) * embedding_dim

        self.embedding = SimpleTabularEmbedding(categories=categories, embedding_dim=embedding_dim)

        # create MLP blocks and output layer
        block_features = [in_features] + block_out_features
        self.blocks = nn.Sequential(OrderedDict([
            (str(i), MLPBlock(in_features=block_features[i - 1],
                              out_features=block_features[i],
                              dropout_prob=dropout_prob)) for i in range(1, len(block_features))
        ]))
        self.output = nn.Linear(in_features=block_features[-1], out_features=out_features)

    def forward(self, x_num: Optional[torch.FloatTensor], x_cat: Optional[torch.IntTensor]):
        """
        :param x_num: [batch_size x in_features_num] the numerical features of this batch
        :param x_cat: [batch_size x len(categories)] the categorical features of this batch
        :return: [batch_size x out_features] the output features
        """
        x = self.embedding(x_num, x_cat)
        x = self.blocks(x)
        x = self.output(x)
        x = x.squeeze(dim=-1)

        return x
