import math

import torch
import torch.nn.functional as F
from torch.nn import Embedding
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ModuleDict
from torch.nn import ModuleList
from torch.nn import Parameter
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer

from torch_geometric.nn import BatchNorm
from torch_geometric.nn import HeteroConv
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import softmax


class PositionalEncoding(Module):
    # adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x


class GumbelTransformerConv(TransformerConv):
    def __init__(self, in_channels, out_channels, heads, **kwargs):
        super().__init__(in_channels, out_channels, heads=heads, **kwargs)
        self.tau = Parameter(torch.rand(1, 1))
        self.gumbel_alpha = 1.0

    def message(self, query_i, key_j, value_j, edge_attr, index, ptr, size_i):
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)

        if self.training:
            # add temperature and gumbel randomness
            gumbels = (
                -torch.empty_like(alpha, memory_format=torch.legacy_contiguous_format)
                .exponential_()
                .log()
            )
            alpha = (alpha + gumbels * self.gumbel_alpha) / self.tau
        else:
            alpha = alpha / self.tau

        alpha = softmax(alpha, index, ptr, size_i)

        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out


class SequenceEncoder(Module):
    def __init__(self, embedding_dim, dim_ff, n_heads, n_layers, dropout=0.1):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=dim_ff,
            dropout=dropout,
        )

        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=LayerNorm(embedding_dim),
        )
        self.bn = BatchNorm(embedding_dim, allow_single_element=True)

    def forward(self, x, src_mask):
        x = self.encoder(x, src_key_padding_mask=src_mask)
        x = torch.select(x, dim=1, index=0)
        x = F.leaky_relu(self.bn(x))
        return x


class GNN(Module):
    def __init__(
        self,
        vocab_size: int = 45,
        embedding_dim: int = 128,
        dim_ff_enc_pep: int = 128,
        n_heads_enc_pep: int = 8,
        n_layers_enc_pep: int = 6,
        dim_ff_enc_mhc: int = 1024,
        n_heads_enc_mhc: int = 8,
        n_layers_enc_mhc: int = 6,
        n_heads_conv: int = 8,
        n_layers_conv: int = 2,
        dim_out_conv: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.act = F.leaky_relu

        dim = n_heads_conv * dim_out_conv

        self.embedding = Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.positional_encoding = PositionalEncoding(embedding_dim)

        # Peptide
        self.encoder = ModuleDict(
            {
                "peptide": ModuleList(
                    [
                        SequenceEncoder(
                            embedding_dim,
                            dim_ff_enc_pep,
                            n_heads_enc_pep,
                            n_layers_enc_pep,
                            dropout,
                        ),
                        Linear(embedding_dim, dim),
                    ]
                ),
                "mhc": ModuleList(
                    [
                        SequenceEncoder(
                            embedding_dim,
                            dim_ff_enc_mhc,
                            n_heads_enc_mhc,
                            n_layers_enc_mhc,
                            dropout,
                        ),
                        Linear(embedding_dim, dim * 2),
                    ]
                ),
            }
        )
        self.bn_conv = ModuleList(
            [
                ModuleDict(
                    {
                        "peptide": BatchNorm(dim, allow_single_element=True),
                        "mhc": BatchNorm(dim * 2, allow_single_element=True),
                    }
                )
                for _ in range(n_layers_conv + 1)
            ]
        )

        self.conv = ModuleList(
            [
                HeteroConv(
                    {
                        ("peptide", "determines", "mhc"): GumbelTransformerConv(
                            (dim, dim * 2),
                            dim_out_conv,
                            heads=n_heads_conv,
                            dropout=dropout,
                        ),
                        ("mhc", "influences", "mhc"): GumbelTransformerConv(
                            dim * 2, dim_out_conv, heads=n_heads_conv, dropout=dropout
                        ),
                        ("peptide", "influences", "peptide"): TransformerConv(
                            dim, dim_out_conv, heads=n_heads_conv, dropout=dropout
                        ),
                    },
                    aggr="cat",
                )
                for _ in range(n_layers_conv)
            ]
        )

        self.fc = Linear(dim * 2, 1)

    def forward(self, batch):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict

        # peptide & mhc encoding
        for nodes in ["peptide", "mhc"]:
            x = x_dict[nodes]
            src_mask = x == 0
            x = self.embedding(x)
            x = self.positional_encoding(x)
            x = self.encoder[nodes][0](x, src_mask)  # encoder
            x_dict[nodes] = self.encoder[nodes][1](x)  # linear

        # conv
        for bn, conv in zip(self.bn_conv[:-1], self.conv, strict=True):
            x_res = x_dict.copy()
            x_dict = {k: self.act(bn[k](v)) for k, v in x_dict.items()}
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: v + x_res[k] for k, v in x_dict.items()}

        x_dict = {k: self.act(self.bn_conv[-1][k](v)) for k, v in x_dict.items()}

        # dense out
        x = torch.sigmoid(self.fc(x_dict["mhc"]))
        return x
