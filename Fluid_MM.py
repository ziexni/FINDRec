import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, layer_norm_eps):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.w_1 = nn.Linear(d_model, d_ff * 2)
        self.w_2 = nn.Linear(d_ff * 2, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_scale = nn.Parameter(torch.ones(1, 1, d_model) * 0.1)
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return residual + self.layer_scale * x


class MambaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        mamba_config = config['mamba']
        self.num_layers = mamba_config['num_layers']
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=mamba_config['hidden_dim'],
                d_state=mamba_config['d_state'],
                d_conv=mamba_config['d_conv'],
                expand=mamba_config['expand']
            ) for _ in range(self.num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(mamba_config['hidden_dim'], eps=mamba_config['norm_eps'])
            for _ in range(self.num_layers)
        ])
        self.dropout = nn.Dropout(config['dropout_prob'])
        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, mamba_config['hidden_dim']) * 0.1)
            for _ in range(self.num_layers)
        ])

    def forward(self, x):
        for i in range(self.num_layers):
            residual = x
            x = self.norms[i](x)
            x = self.mamba_layers[i](x)
            x = self.dropout(x)
            x = residual + self.layer_scales[i] * x
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config['num_attention_heads']
        self.hidden_size = config['multimodal']['hidden_size']
        self.head_dim = self.hidden_size // self.num_heads

        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.image_projection = nn.Sequential(
            nn.Linear(config['image']['projection_dim'], self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(config['dropout_prob'])
        )
        self.text_projection = nn.Sequential(
            nn.Linear(config['text']['projection_dim'], self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(config['dropout_prob'])
        )
        self.image_to_text_heads = nn.ModuleList([
            nn.MultiheadAttention(self.head_dim, 1, dropout=config['dropout_prob'])
            for _ in range(self.num_heads)
        ])
        self.text_to_image_heads = nn.ModuleList([
            nn.MultiheadAttention(self.head_dim, 1, dropout=config['dropout_prob'])
            for _ in range(self.num_heads)
        ])
        self.head_norms = nn.ModuleList([
            nn.LayerNorm(self.head_dim * 2)
            for _ in range(self.num_heads)
        ])
        self.scaling = float(self.head_dim) ** -0.5

    def forward(self, image_features, text_features, attention_mask=None):
        batch_size, seq_len = image_features.size(0), image_features.size(1)
        image_proj = self.image_projection(image_features)
        text_proj = self.text_projection(text_features)
        image_proj = F.normalize(image_proj, p=2, dim=-1)
        text_proj = F.normalize(text_proj, p=2, dim=-1)

        image_heads = image_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        text_heads = text_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        key_padding_mask = (~attention_mask).bool() if attention_mask is not None else None

        head_outputs = []
        for head_idx in range(self.num_heads):
            curr_image = (image_heads[:, head_idx] * self.scaling).transpose(0, 1)
            curr_text  = (text_heads[:, head_idx]  * self.scaling).transpose(0, 1)

            img2text, _ = self.image_to_text_heads[head_idx](
                curr_image, curr_text, curr_text, key_padding_mask=key_padding_mask)
            text2img, _ = self.text_to_image_heads[head_idx](
                curr_text, curr_image, curr_image, key_padding_mask=key_padding_mask)

            img2text = curr_image + img2text.transpose(0, 1) if True else img2text
            # transpose back
            img2text = img2text.transpose(0, 1) if img2text.shape[0] != batch_size else img2text
            text2img = text2img.transpose(0, 1) if text2img.shape[0] != batch_size else text2img

            # 이미 transpose된 경우 재조정
            curr_image_b = image_heads[:, head_idx]  # [B, L, D]
            curr_text_b  = text_heads[:, head_idx]

            img2text_out, _ = self.image_to_text_heads[head_idx](
                curr_image_b.transpose(0,1), curr_text_b.transpose(0,1),
                curr_text_b.transpose(0,1), key_padding_mask=key_padding_mask)
            text2img_out, _ = self.text_to_image_heads[head_idx](
                curr_text_b.transpose(0,1), curr_image_b.transpose(0,1),
                curr_image_b.transpose(0,1), key_padding_mask=key_padding_mask)

            img2text_out = curr_image_b + img2text_out.transpose(0, 1)
            text2img_out = curr_text_b  + text2img_out.transpose(0, 1)

            combined = torch.cat([img2text_out, text2img_out], dim=-1)
            head_outputs.append(self.head_norms[head_idx](combined))

        return torch.stack(head_outputs, dim=1)  # [B, num_heads, L, head_dim*2]
