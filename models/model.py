import math
from typing import Optional, List, Literal

import torch
from torch import nn
import torch.nn.functional as F

from .encoding import PeriodicEncoding, PiecewiseLinearEncoding, LinearNumericalEncoding
from .ft_transformer import Transformer


class BilinearAttention(nn.Module):
    def __init__(self, d_embedding: int):
        super().__init__()
        self.bilinear = nn.Bilinear(
            in1_features=d_embedding,
            in2_features=d_embedding,
            out_features=1,
            bias=True)

    def forward(self, hidden_states: torch.Tensor, target: torch.Tensor):
        """
        :param hidden_states: [batch_size x n_features x d_embedding] embeddings without target embedding
        :param target: [batch_size x 1 x d_embedding]
        :return: [batch_size x d_embedding] the new representation
        """
        batch_size, n_features, _ = hidden_states.size()

        target = target.repeat(1, n_features, 1)
        att_scores = self.bilinear(hidden_states, target).squeeze(dim=-1)
        att_scores = F.tanh(att_scores)
        att_scores = F.softmax(att_scores, dim=-1)

        return torch.einsum('bi,bij->bj', att_scores, hidden_states)


def create_initial_representations(x_num: Optional[torch.Tensor],
                                   x_cat: Optional[torch.Tensor],
                                   input_num: Optional[nn.Module],
                                   input_cat: Optional[nn.Embedding],
                                   category_offsets: torch.Tensor,
                                   cat_bias: Optional[nn.Parameter]):
    assert (x_num is None) == (input_num is None), 'x_num and input_num should both be None or not None'
    assert (x_cat is None) == (input_cat is None), 'x_cat and input_cat should both be None or not None'
    assert not (x_num is None and x_cat is None), 'expected at least x_num or x_cat'

    x_nonnull: torch.Tensor = x_num if x_num is not None else x_cat
    batch_size, _ = x_nonnull.size()

    x_temp: List[torch.Tensor] = [
        input_cat(torch.zeros(batch_size, 1, device=x_nonnull.device, dtype=torch.int32))]

    if input_num is not None:
        x_num = input_num(x_num)
        x_temp.append(x_num)

    if input_cat is not None:
        x_cat = input_cat(x_cat + category_offsets)

        if cat_bias is not None:
            x_cat += cat_bias

        x_temp.append(x_cat)

    x = torch.cat(x_temp, dim=-2)
    return x


class Model(nn.Module):
    def __init__(self,
                 categories: List[int],
                 d_numerical: int,
                 d_token: int,
                 d_out: int,
                 n_layers: int,
                 n_heads: int,
                 attention_dropout: float,
                 ffn_dropout: float,
                 d_ffn_factor: float,
                 input_num: Optional[PeriodicEncoding | PiecewiseLinearEncoding | LinearNumericalEncoding] = None,
                 bilinear=True):
        super().__init__()
        self.d_numerical = d_numerical
        self.d_token = d_token
        self.categories = categories

        self.input_num = input_num

        if self.input_num is None and d_numerical > 0:
            self.input_num = LinearNumericalEncoding(in_features=d_numerical, d_token=d_token)

        self.input_cat: Optional[nn.Embedding] = None
        self.cat_bias: Optional[nn.Parameter] = None

        if len(categories) > 0:
            self.register_buffer('category_offsets', torch.tensor([0] + categories[:-1], dtype=torch.int32).cumsum(0))
            self.input_cat = nn.Embedding(num_embeddings=sum(categories), embedding_dim=d_token)
            use_feature_tokenizer = isinstance(input_num, LinearNumericalEncoding)

            if use_feature_tokenizer:
                self.cat_bias = nn.Parameter(torch.randn(len(categories), d_token))

        assert self.input_num is not None or self.input_cat is not None, 'Expected at least one numerical or categorical input layer'

        self.transformer = Transformer(d_embedding=d_token, n_layers=n_layers, n_heads=n_heads,
                                       d_head=d_token // n_heads, attn_dropout=attention_dropout,
                                       d_feedforward=int(d_ffn_factor * d_token), ffn_dropout=ffn_dropout)

        self.bilinear = None
        if bilinear:
            self.bilinear = BilinearAttention(d_embedding=d_token)

        d_output_hidden = 2 * d_token if self.bilinear is not None else d_token
        self.output = nn.Sequential(
            nn.LayerNorm(d_output_hidden),
            nn.ReLU(),
            nn.Linear(d_output_hidden, d_out)
        )

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor], output_hidden_states=False):
        hidden_states = create_initial_representations(x_num, x_cat, self.input_num, self.input_cat,
                                                       self.get_buffer('category_offsets'), self.cat_bias)
        hidden_states = self.transformer(hidden_states)

        batch_size, n_features, d_embedding = hidden_states.size()
        cls_embedding, x = hidden_states.split([1, n_features - 1], dim=-2)

        if self.bilinear is not None:
            x = self.bilinear(x, cls_embedding)
            x = torch.cat((cls_embedding.squeeze(dim=-2), x), dim=-1)
        else:
            # pick [CLS] embedding
            x = x[:, 0, :]

        x = self.output(x).squeeze(dim=-1)

        if output_hidden_states:
            return x, hidden_states

        return x


class ModelForMaskedLM(nn.Module):
    def __init__(self, model: Model):
        super().__init__()
        self.d_token = model.d_token
        categories = model.categories

        self.register_buffer('category_offsets', model.get_buffer('category_offsets'))

        self.input_cat = model.input_cat
        self.cat_bias = model.cat_bias
        self.input_num = model.input_num
        self.transformer = model.transformer
        self.transform_hidden_state = nn.Sequential(
            nn.Linear(self.d_token, self.d_token),
            nn.ReLU(),
            nn.LayerNorm(self.d_token)
        )
        self.num_weight = nn.Parameter(torch.empty(model.d_numerical, self.d_token))
        # Each column has its own vocabulary size because one column may not obtain values from another column.
        self.cat_decoders = nn.ModuleList(
            [nn.Linear(self.d_token, vocab_size, bias=False) for vocab_size in categories])

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.num_weight, a=math.sqrt(5))

    def forward(self,
                x_num: Optional[torch.Tensor],
                x_cat: Optional[torch.Tensor],
                x_num_mask: Optional[torch.Tensor],
                x_cat_mask: Optional[torch.Tensor]):
        """
        :param x_num: [batch_size x d_numerical] the numerical features of this batch
        :param x_cat: [batch_size x d_categorical] the categorical features for this batch
        :param x_num_mask: [batch_size x d_numerical] True if feature is masked, False otherwise
        :param x_cat_mask: [batch_size x d_categorical] True if the feature is masked, False otherwise
        :return: (x_num, x_cat) the predicted features for numerical and categorical features
        """
        assert (x_num is None) == (self.input_num is None), 'x_num and input_num should both be None or not None'
        assert (x_cat is None) == (self.input_cat is None), 'x_cat and input_cat should both be None or not None'
        assert not (x_num is None and x_cat is None), 'expected at least x_num or x_cat'

        x_nonnull: torch.Tensor = x_num if x_num is not None else x_cat
        batch_size, _ = x_nonnull.size()
        d_numerical, embedding_dim = self.num_weight.size()
        d_cat = len(self.cat_decoders)

        x_num_actual = x_num
        x_cat_actual = x_cat

        hidden_state = create_initial_representations(x_num, x_cat, self.input_num, self.input_cat,
                                                      self.get_buffer('category_offsets'), self.cat_bias)
        cls_mask = torch.zeros(batch_size, 1, device=hidden_state.device)
        hidden_state = torch.einsum('bij,bi->bij',
                                    hidden_state,
                                    1 - torch.cat((cls_mask, x_num_mask, x_cat_mask), dim=-1))

        hidden_state = self.transformer(hidden_state)
        hidden_state = self.transform_hidden_state(hidden_state)

        cls, hidden_states_num, hidden_states_cat = hidden_state.split([1, d_numerical, d_cat], dim=-2)
        hidden_states_num = torch.einsum('bij,ij->bi', hidden_states_num, self.num_weight)

        num_loss_fn = nn.MSELoss(reduction='none')
        num_loss_sum = torch.sum(num_loss_fn(hidden_states_num, x_num_actual) * x_num_mask)

        hidden_states_cat = hidden_states_cat.transpose(-2, -3)
        cat_loss_fn = nn.CrossEntropyLoss(reduction='none')
        x_cat_actual_transpose = torch.transpose(x_cat_actual, -1, -2)
        cat_loss_sum = torch.stack(
            [cat_loss_fn(F.softmax(decoder(hidden_states_cat[i]), dim=-1), x_cat_actual_transpose[i]) for i, decoder in
             enumerate(self.cat_decoders)])
        cat_loss_sum = torch.transpose(cat_loss_sum, -1, -2)
        cat_loss_sum = (cat_loss_sum * x_cat_mask).sum(dim=-1)
        cat_loss_sum = cat_loss_sum.sum()

        n_masked = torch.sum(x_num_mask) + torch.sum(x_cat_mask)
        loss = (num_loss_sum + cat_loss_sum) / n_masked

        if torch.isnan(loss).item():
            raise ValueError("MaskedLM loss is nan")

        return loss
