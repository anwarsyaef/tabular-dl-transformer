from collections import OrderedDict
from typing import List, Optional

import torch
from torch import nn

from .mlp import SimpleTabularEmbedding


class ResNetBlock(nn.Module):
    def __init__(self, d: int, d_hidden: int, hidden_dropout: Optional[float] = None,
                 residual_dropout: Optional[float] = None):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(num_features=d),
            nn.Linear(in_features=d, out_features=d_hidden),
            nn.ReLU(),
            nn.Dropout(p=hidden_dropout),
            nn.Linear(in_features=d_hidden, out_features=d),
            nn.Dropout(p=residual_dropout)
        )

    def forward(self, x: torch.Tensor):
        return x + self.layer(x)


class ResNet(nn.Module):
    def __init__(self, *,
                 in_features_num: int,
                 categories: Optional[List[int]],
                 d_embedding: int,
                 d: int,
                 d_hidden_factor: float,
                 n_layers: int,
                 hidden_dropout: float,
                 residual_dropout: float,
                 out_features: int):
        super().__init__()
        self.embedding = SimpleTabularEmbedding(categories=categories, embedding_dim=d_embedding)
        self.input_linear = nn.Linear(in_features=in_features_num + len(categories) * d_embedding, out_features=d)
        self.blocks = nn.Sequential(OrderedDict([
            (str(i),
             ResNetBlock(d=d,
                         d_hidden=int(d * d_hidden_factor),
                         hidden_dropout=hidden_dropout,
                         residual_dropout=residual_dropout)) for i in range(n_layers)
        ]))
        self.output = nn.Sequential(
            nn.BatchNorm1d(num_features=d),
            nn.ReLU(),
            nn.Linear(in_features=d, out_features=out_features)
        )

    def forward(self, x_num: Optional[torch.FloatTensor], x_cat: Optional[torch.IntTensor]):
        x = self.embedding(x_num, x_cat)
        x = self.input_linear(x)
        x = self.blocks(x)
        x = self.output(x)
        x = x.squeeze(dim=-1)

        return x

# class ResNet(nn.Module):
#     def __init__(
#             self,
#             *,
#             d_numerical: int,
#             categories: Optional[List[int]],
#             d_embedding: int,
#             d: int,
#             d_hidden_factor: float,
#             n_layers: int,
#             # activation: str,
#             # normalization: str,
#             hidden_dropout: float,
#             residual_dropout: float,
#             out_features: int,
#     ) -> None:
#         super().__init__()
#
#         self.main_activation = nn.ReLU()
#         self.last_activation = nn.ReLU()
#         self.residual_dropout = residual_dropout
#         self.hidden_dropout = hidden_dropout
#
#         d_in = d_numerical
#         d_hidden = int(d * d_hidden_factor)
#
#         if categories is not None:
#             d_in += len(categories) * d_embedding
#             category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
#             self.register_buffer('category_offsets', category_offsets)
#             self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
#             nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
#             print(f'{self.category_embeddings.weight.shape=}')
#
#         self.first_layer = nn.Linear(d_in, d)
#         self.layers = nn.ModuleList(
#             [
#                 nn.ModuleDict(
#                     {
#                         'norm': nn.BatchNorm1d(d),
#                         'linear0': nn.Linear(d, d_hidden),
#                         'linear1': nn.Linear(d_hidden, d),
#                     }
#                 )
#                 for _ in range(n_layers)
#             ]
#         )
#         self.last_normalization = nn.BatchNorm1d(d)
#         self.head = nn.Linear(d, out_features)
#
#     def forward(self, x_num: torch.Tensor, x_cat: Optional[torch.Tensor]) -> torch.Tensor:
#         x = []
#         if x_num is not None:
#             x.append(x_num)
#         if x_cat is not None:
#             x.append(
#                 self.category_embeddings(x_cat + self.category_offsets[None]).view(
#                     x_cat.size(0), -1
#                 )
#             )
#         x = torch.cat(x, dim=-1)
#
#         x = self.first_layer(x)
#         for layer in self.layers:
#             layer = cast(Dict[str, nn.Module], layer)
#             z = x
#             z = layer['norm'](z)
#             z = layer['linear0'](z)
#             z = self.main_activation(z)
#             if self.hidden_dropout:
#                 z = F.dropout(z, self.hidden_dropout, self.training)
#             z = layer['linear1'](z)
#             if self.residual_dropout:
#                 z = F.dropout(z, self.residual_dropout, self.training)
#             x = x + z
#         x = self.last_normalization(x)
#         x = self.last_activation(x)
#         x = self.head(x)
#         x = x.squeeze(-1)
#         return x
