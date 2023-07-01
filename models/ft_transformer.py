from typing import Optional

import torch
from torch import nn
from einops import rearrange, einsum


class FeatureTokenizer(nn.Module):
    def __init__(self, categories: list[int], in_features_num: int, embedding_dim: int):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + in_features_num > 0, 'input shape must not be empty'

        self.in_features_cat = len(categories)
        self.in_features_num = in_features_num
        self.cls_num_weight = nn.Parameter(torch.randn(1 + in_features_num, embedding_dim))

        if len(categories) > 0:
            self.register_buffer('category_offsets', torch.tensor([0] + categories[:-1], dtype=torch.int32).cumsum(0))
            self.cat_embedding = nn.Embedding(num_embeddings=sum(categories), embedding_dim=embedding_dim)

        self.bias = nn.Parameter(torch.randn(in_features_num + len(categories), embedding_dim))
        self.cls = nn.Parameter(torch.ones(1, embedding_dim))

    def forward(self, x_num: Optional[torch.Tensor] = None, x_cat: Optional[torch.Tensor] = None):
        """
        :param x_num: [batch_size x in_features_num]
        :param x_cat: [batch_size x in_features_cat]
        :return: [batch_size, 1 + in_features_num + in_features_cat, embedding_dim] the embeddings for the input with [CLS]
        """
        assert x_num is not None and x_cat is not None, 'Expected at least x_num or x_cat'
        x_nonnull = x_num if x_num is not None else x_cat
        batch_size, _ = x_nonnull.size()

        x_num = torch.cat([torch.ones(batch_size, 1, device=x_nonnull.device)] + ([] if x_num is None else [x_num]),
                          dim=-1)
        x_num = torch.einsum('bi,ij->bij', x_num, self.cls_num_weight)
        x_temp: list[torch.Tensor] = [x_num]

        if self.in_features_cat > 0:
            x_cat = self.cat_embedding(x_cat + self.get_buffer('category_offsets'))
            x_temp.append(x_cat)

        x = torch.cat(x_temp, dim=-2)

        _, embedding_dim = self.bias.size()
        bias = torch.cat([torch.zeros(1, embedding_dim, device=x_nonnull.device), self.bias])
        x += bias
        return x


class GEGLU(nn.Module):
    """https://github.com/lucidrains/tab-transformer-pytorch/blob/main/tab_transformer_pytorch/ft_transformer.py"""

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * nn.functional.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, d_embedding: int, d_feedforward: int, dropout=0.):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(d_embedding),
            nn.Linear(d_embedding, d_feedforward * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(d_feedforward, d_embedding)
        )

    def forward(self, x: torch.Tensor):
        return self.layer(x)


class Attention(nn.Module):
    """https://github.com/lucidrains/tab-transformer-pytorch/blob/main/tab_transformer_pytorch/ft_transformer.py"""

    def __init__(
            self,
            d_embedding: int,
            n_heads=8,
            d_head=64,
            dropout=0.
    ):
        super().__init__()
        inner_dim = d_head * n_heads
        self.heads = n_heads
        self.scale = d_head ** -0.5

        self.norm = nn.LayerNorm(d_embedding)

        self.to_q = nn.Linear(d_embedding, inner_dim, bias=False)
        self.to_k = nn.Linear(d_embedding, inner_dim, bias=False)
        self.to_v = nn.Linear(d_embedding, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, d_embedding, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, only_cls=False):
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x[:, [0], :]) if only_cls else self.to_q(x)
        k, v = self.to_k(x), self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        attn = sim.softmax(dim=-1)
        dropped_attn = self.dropout(attn)

        out = einsum(dropped_attn, v, 'b h i j, b h j d -> b h i d')
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)

        return out, attn


class Transformer(nn.Module):
    """https://github.com/lucidrains/tab-transformer-pytorch/blob/main/tab_transformer_pytorch/ft_transformer.py"""

    def __init__(
            self,
            d_embedding: int,
            n_layers: int,
            n_heads: int,
            d_head: int,
            attn_dropout: float,
            ffn_dropout: float,
            d_feedforward: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                Attention(d_embedding, n_heads=n_heads, d_head=d_head, dropout=attn_dropout),
                FeedForward(d_embedding=d_embedding, d_feedforward=d_feedforward, dropout=ffn_dropout),
            ]))

    def forward(self, x, return_attn=False, only_cls_on_last=False):
        post_softmax_attns = []

        for i, (attn, ff) in enumerate(self.layers):
            is_last_layer = i + 1 == len(self.layers)
            attn_out, post_softmax_attn = attn(x, only_cls=only_cls_on_last and is_last_layer)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x

        return x, post_softmax_attns


class FTTransformer(nn.Module):
    def __init__(self,
                 categories: list[int],
                 d_numerical: int,
                 d_token: int,
                 d_out: int,
                 n_layers: int,
                 n_heads: int,
                 attention_dropout: float,
                 ffn_dropout: float,
                 d_ffn_factor: float,
                 # unused
                 token_bias: bool = True,
                 # transformer
                 residual_dropout: float = 0.,
                 activation: str = 'reglu',
                 prenormalization: bool = True,
                 initialization: str = 'kaiming',
                 kv_compression: Optional[float] = None,
                 kv_compression_sharing: Optional[str] = None,
                 ):
        super().__init__()
        assert d_token % n_heads == 0, 'embedding_dim must be a multiple of the number of heads'

        self.ft = FeatureTokenizer(categories=categories, in_features_num=d_numerical, embedding_dim=d_token)
        self.transformer = Transformer(d_embedding=d_token, n_layers=n_layers, n_heads=n_heads,
                                       d_head=d_token // n_heads,
                                       attn_dropout=attention_dropout, d_feedforward=int(d_ffn_factor * d_token),
                                       ffn_dropout=ffn_dropout)
        self.output = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, d_out)
        )

    def forward(self, x_num: torch.FloatTensor, x_cat: torch.IntTensor, only_cls_on_last=False):
        """
        :param x_num: [batch_size x n_num] a tensor containing numerical features
        :param x_cat: [batch_size x n_cat] a tensor containing categorical features
        :return:
        """
        x = self.ft(x_num, x_cat)
        x = self.transformer(x, only_cls_on_last=only_cls_on_last)

        # pick [CLS] embeddings
        x = x[:, 0, :]
        x = self.output(x)
        x = x.squeeze(dim=-1)

        return x
