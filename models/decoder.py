import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, self_pos, cross_pos, x_mask=None, cross_mask=None):
        # x: [B, L, D], self_pos: [1, L, D], cross_pos: [1, S, D]
        p_x = self_pos.expand(x.size(0), -1, -1)         # [B, L, D]
        p_c = cross_pos.expand(cross.size(0), -1, -1)     # [B, S, D]

        # Self-attention with TUPE-style positional projections
        new_self, _ = self.self_attention(
            x, x, x,
            x_mask,
            p_x,  # pos_queries
            p_x   # pos_keys
        )
        x = x + self.dropout(new_self)
        x = self.norm1(x)

        # Cross-attention with TUPE-style positional projections
        new_cross, _ = self.cross_attention(
            x, cross, cross,
            cross_mask,
            p_x,  # query positions
            p_c   # key positions
        )
        x = x + self.dropout(new_cross)
        y = self.norm2(x)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, self_pos, cross_pos, x_mask=None, cross_mask=None):
        # x: [B, L, D], cross: [B, S, D]
        # self_pos: [1, L, D], cross_pos: [1, S, D]

        for layer in self.layers:
            x = layer(
                x, cross,
                self_pos=self_pos,
                cross_pos=cross_pos,
                x_mask=x_mask,
                cross_mask=cross_mask
            )

        if self.norm is not None:
            x = self.norm(x)

        return x
