import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):  # x: [S, B, E]
        return self.dropout(x + self.pe[:x.size(0)])


def _split_heads(h, attn: nn.MultiheadAttention, num_heads, head_dim):
    """h [S, bsz, E] -> q, k, v [bsz, heads, S, head_dim]."""
    S, bsz, _ = h.shape
    q, k, v = F.linear(h, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
    return [t.view(S, bsz, num_heads, head_dim).permute(1, 2, 0, 3) for t in (q, k, v)]


def _merge_heads(z, attn: nn.MultiheadAttention):
    """z [bsz, heads, S, head_dim] -> out_proj(concat) [S, bsz, E]."""
    bsz, H, S, d = z.shape
    return F.linear(z.permute(2, 0, 1, 3).contiguous().view(S, bsz, H * d), attn.out_proj.weight, attn.out_proj.bias)


class SpatialTransformerLayer(nn.Module):
    """Pre-LN transformer layer over the markers of one frame; x is [N, B*L, E]."""
    def __init__(self, spatial_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = spatial_dim // num_heads
        self.ln1 = nn.LayerNorm(spatial_dim)
        self.self_attn = nn.MultiheadAttention(spatial_dim, num_heads, dropout=dropout, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(spatial_dim)
        self.ffn = nn.Sequential(
            nn.Linear(spatial_dim, spatial_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_dim * 4, spatial_dim),
        )
        self._capture_per_head = False
        self.per_head_output = None      # [B*L, heads, N, head_dim] when captured
        self._capture_residual_stream = False
        self.ffn_delta = None            # [N, B*L, E] when captured

    def forward(self, x, need_attn=False, average_attn_weights=False):
        h = self.ln1(x)
        if self._capture_per_head:
            attn_out, attn_w = self._forward_capture_heads(h)
        else:
            attn_out, attn_w = self.self_attn(h, h, h, need_weights=need_attn, average_attn_weights=average_attn_weights)
        x = x + self.dropout(attn_out)
        ffn_d = self.ffn(self.ln2(x))
        if self._capture_residual_stream:
            self.ffn_delta = ffn_d.detach().cpu()
        return x + ffn_d, attn_out, attn_w

    def _forward_capture_heads(self, h):
        """Attention computed per head; stores the per-head output before out_proj."""
        q, k, v = _split_heads(h, self.self_attn, self.num_heads, self.head_dim)
        z = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        self.per_head_output = z.detach().cpu()
        return _merge_heads(z, self.self_attn), None


class SpatialTransformer(nn.Module):
    def __init__(self, marker_dim, spatial_dim, num_heads, dropout, num_markers=None, num_layers=1):
        super().__init__()
        self.input_projection = nn.Linear(marker_dim, spatial_dim)
        self.marker_embed = nn.Embedding(num_markers, spatial_dim) if num_markers is not None else None
        self.layers = nn.ModuleList([SpatialTransformerLayer(spatial_dim, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x):
        B, L, N, D = x.size()                                   # x: [B, L, N, D]
        x = self.input_projection(x.view(B * L, N, D).transpose(0, 1).contiguous())   # [N, B*L, E]
        if self.marker_embed is not None:
            x = x + self.marker_embed(torch.arange(N, device=x.device)).unsqueeze(1)
        for layer in self.layers:
            x = layer(x)[0]
        return x.transpose(0, 1).contiguous().view(B, L, N, -1)

    def enable_per_head_capture(self):
        for layer in self.layers:
            layer._capture_per_head = True

    def disable_per_head_capture(self):
        for layer in self.layers:
            layer._capture_per_head = False
            layer.per_head_output = None

    def enable_residual_capture(self):
        for layer in self.layers:
            layer._capture_residual_stream = True

    def disable_residual_capture(self):
        for layer in self.layers:
            layer._capture_residual_stream = False
            layer.ffn_delta = None


class TemporalTransformerLayer(nn.Module):
    """Pre-LN transformer layer over the frames of one marker; x is [L, N*B, E]."""
    def __init__(self, temporal_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = temporal_dim // num_heads
        self.ln1 = nn.LayerNorm(temporal_dim)
        self.self_attn = nn.MultiheadAttention(temporal_dim, num_heads, dropout=dropout, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(temporal_dim)
        self.ffn_linear1 = nn.Linear(temporal_dim, temporal_dim * 4)
        self.ffn_act = nn.GELU()
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_linear2 = nn.Linear(temporal_dim * 4, temporal_dim)
        self.ffn = nn.Sequential(self.ffn_linear1, self.ffn_act, self.ffn_dropout, self.ffn_linear2)
        self._capture_per_head = False
        self.per_head_output = None      # [N*B, heads, L, head_dim] when captured
        self._capture_mlp_probes = False
        self.mlp_delta = None            # [L, N*B, E] when captured

    def forward(self, x, need_attn=False, average_attn_weights=False):
        h = self.ln1(x)
        if self._capture_per_head:
            attn_out, attn_w = self._forward_capture_heads(h)
        else:
            attn_out, attn_w = self.self_attn(h, h, h, need_weights=need_attn, average_attn_weights=average_attn_weights)
        x = x + self.dropout(attn_out)
        delta = self.ffn(self.ln2(x))
        if self._capture_mlp_probes:
            self.mlp_delta = delta.detach().cpu()
        return x + delta, attn_out, attn_w

    def _forward_capture_heads(self, h):
        """Attention computed per head; stores the per-head output before out_proj."""
        q, k, v = _split_heads(h, self.self_attn, self.num_heads, self.head_dim)
        z = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        self.per_head_output = z.detach().cpu()
        return _merge_heads(z, self.self_attn), None

    def _forward_qkov_split(self, h, frozen_z=None, capture=False):
        """Attention with explicit softmax(QK^T)V. frozen_z [bsz, heads, L, head_dim] replaces the
        per-head output z before out_proj. With capture, also returns the attention weights and V."""
        q, k, v = _split_heads(h, self.self_attn, self.num_heads, self.head_dim)
        attn_w = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.head_dim ** -0.5, dim=-1)
        z = torch.matmul(attn_w, v) if frozen_z is None else frozen_z.to(v.device)
        attn_out = _merge_heads(z, self.self_attn)
        if not capture:
            return attn_out, None, None
        self.per_head_output = z.detach().cpu()
        return attn_out, attn_w.detach(), v.detach()


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, temporal_dim, num_heads, dropout, num_layers=1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, temporal_dim)
        self.positional_encoding = PositionalEncoding(temporal_dim, dropout)
        self.layers = nn.ModuleList([TemporalTransformerLayer(temporal_dim, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x):
        B, L, N, E = x.size()                                   # x: [B, L, N, E]
        x = x.permute(2, 0, 1, 3).contiguous().view(N * B, L, E).transpose(0, 1)   # [L, N*B, E]
        x = self.positional_encoding(self.input_projection(x))
        for layer in self.layers:
            x = layer(x)[0]
        return x.transpose(0, 1).contiguous().view(N, B, L, -1).permute(1, 2, 0, 3)

    def enable_per_head_capture(self):
        for layer in self.layers:
            layer._capture_per_head = True

    def disable_per_head_capture(self):
        for layer in self.layers:
            layer._capture_per_head = False
            layer.per_head_output = None

    def enable_mlp_probe_capture(self):
        for layer in self.layers:
            layer._capture_mlp_probes = True

    def disable_mlp_probe_capture(self):
        for layer in self.layers:
            layer._capture_mlp_probes = False
            layer.mlp_delta = None


class SpatialTemporalTransformer(nn.Module):
    def __init__(self, marker_dim, spatial_dim, temporal_dim, spatial_num_heads, temporal_num_heads,
                 fc_dim, output_dim, dropout, num_markers=None, agg_num_heads=4,
                 num_spatial_layers=1, num_temporal_layers=1):
        super().__init__()
        self.output_dim = output_dim
        self.spatial_transformer = SpatialTransformer(marker_dim, spatial_dim, spatial_num_heads, dropout,
                                                      num_markers=num_markers, num_layers=num_spatial_layers)
        self.temporal_transformer = TemporalTransformer(spatial_dim, temporal_dim, temporal_num_heads, dropout,
                                                        num_layers=num_temporal_layers)
        self.agg_attn = nn.MultiheadAttention(embed_dim=temporal_dim, num_heads=agg_num_heads, dropout=dropout, batch_first=False)
        self.agg_query_token = nn.Parameter(torch.randn(1, 1, temporal_dim))
        self.fc = nn.Sequential(
            nn.Linear(temporal_dim, fc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, output_dim)
        )


class BaselineMeanPoolAgg(SpatialTemporalTransformer):
    """ST(Flat): spatial transformer -> temporal transformer -> mean over markers -> FC.
    x [B, L, N, 3] -> [B, L] (output_dim 1)."""

    def forward(self, x):
        out = self.fc(self.temporal_transformer(self.spatial_transformer(x)).mean(dim=2))
        return out.squeeze(-1) if self.output_dim == 1 else out
