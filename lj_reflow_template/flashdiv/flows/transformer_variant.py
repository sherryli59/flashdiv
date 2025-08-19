import math
import torch
import torch.nn as nn
from flashdiv.flows.flow_net_torchdiffeq import FlowNet
from flashdiv.flows.transformer import EncoderLayer, TimestepEmbedder


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for 1D sequences."""

    def __init__(self, seq_length: int, d_model: int):
        super().__init__()
        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to the input tensor."""
        return x + self.pe


class TransformerVariant(FlowNet):
    """Transformer tailored for inputs with shape ``[batch_size, dim]``.

    Each dimension is treated as a token. Positional encodings break the
    permutational symmetry among dimensions.
    """

    def __init__(
        self,
        seq_length: int,
        n_layers: int = 4,
        n_head: int = 4,
        d_k: int = 128,
        d_v: int = 128,
        d_hidden: int = 128,
        dropout: float = 0.0,
        time_emb: bool = True,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.dropout = nn.Dropout(p=dropout)
        self.affine_in = nn.Linear(1, d_hidden)
        self.pos_encoding = PositionalEncoding(seq_length, d_hidden)
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(
                    d_hidden,
                    d_hidden,
                    n_head,
                    d_k,
                    d_v,
                    dropout=dropout,
                    is_final_layer=(i == n_layers),
                )
                for i in range(n_layers + 1)
            ]
        )
        self.affine_out = nn.Linear(d_hidden, 1)
        if time_emb:
            self.time_emb = TimestepEmbedder(hidden_size=d_hidden)
        self.initialize_weights()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_attns: bool = False,
    ):
        # reshape to (batch, seq_length, 1)
        x = x.unsqueeze(-1)
        x = self.affine_in(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        if t is not None:
            if t.dim() == 0:
                t = t * torch.ones(x.shape[0], device=x.device)
            time_emb = self.time_emb(t)
        else:
            time_emb = None

        enc_slf_attn_list = []
        for enc_layer in self.layer_stack:
            x, enc_slf_attn = enc_layer(x, context=time_emb, slf_attn_mask=mask)
            if return_attns:
                enc_slf_attn_list.append(enc_slf_attn)

        x = self.affine_out(x).squeeze(-1)
        if return_attns:
            return x, enc_slf_attn_list
        return x

    def initialize_weights(self) -> None:
        """Initialize transformer weights following ``transformer.py``."""

        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        if hasattr(self, "time_emb"):
            nn.init.normal_(self.time_emb.mlp[0].weight, std=0.02)
            nn.init.normal_(self.time_emb.mlp[2].weight, std=0.02)

        for layer in self.layer_stack:
            nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.affine_out.weight, 0)
        nn.init.constant_(self.affine_out.bias, 0)
