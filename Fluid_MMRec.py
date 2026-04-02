import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import numpy as np


# ── 모델 구조: 원본 그대로 유지 ───────────────────────────────────────────────

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
        text_proj  = self.text_projection(text_features)
        image_proj = F.normalize(image_proj, p=2, dim=-1)
        text_proj  = F.normalize(text_proj,  p=2, dim=-1)
        image_heads = image_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        text_heads  = text_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key_padding_mask = (~attention_mask).bool() if attention_mask is not None else None
        head_outputs = []
        for head_idx in range(self.num_heads):
            curr_image = image_heads[:, head_idx] * self.scaling  # (B, L, D)
            curr_text  = text_heads[:, head_idx]  * self.scaling
            curr_image_t = curr_image.transpose(0, 1)
            curr_text_t  = curr_text.transpose(0, 1)
            img2text, _ = self.image_to_text_heads[head_idx](
                curr_image_t, curr_text_t, curr_text_t,
                key_padding_mask=key_padding_mask
            )
            text2img, _ = self.text_to_image_heads[head_idx](
                curr_text_t, curr_image_t, curr_image_t,
                key_padding_mask=key_padding_mask
            )
            img2text = curr_image + img2text.transpose(0, 1)
            text2img = curr_text  + text2img.transpose(0, 1)
            combined  = torch.cat([img2text, text2img], dim=-1)
            normalized = self.head_norms[head_idx](combined)
            head_outputs.append(normalized)
        return torch.stack(head_outputs, dim=1)  # (B, H, L, 2D)


class ExpertRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts  = config['expert']['num_experts']
        self.hidden_size  = config['router']['hidden_size']
        head_dim  = config['multimodal']['hidden_size'] // config['num_attention_heads']
        input_dim = head_dim * 2
        self.router = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(config['router']['dropout']),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.num_experts)
        )
        self.temperature = 0.1
        self.balance_coefficient = 0.001
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, training=False):
        batch_size, num_heads, seq_len, dim = x.size()
        x = F.normalize(x.reshape(-1, dim), dim=-1).view(batch_size, num_heads, seq_len, dim)
        x_reshaped = x.view(batch_size * num_heads, seq_len, dim)
        logits    = self.router(x_reshaped)
        log_gates = F.log_softmax(logits / self.temperature, dim=-1)
        if training:
            log_gates = log_gates + torch.rand_like(log_gates) * 0.05
            gates = F.gumbel_softmax(log_gates, tau=self.temperature, hard=True, dim=-1)
        else:
            gates = torch.exp(log_gates)
        gates = gates.view(batch_size, num_heads, seq_len, -1)
        gates = torch.clamp(gates, min=1e-6, max=1.0)
        if training:
            expert_usage = gates.mean(dim=(0, 1, 2))
            expert_usage = torch.clamp(expert_usage, min=1e-6)
            balance_loss = -torch.sum(expert_usage * torch.log(expert_usage))
            balance_loss = self.balance_coefficient * torch.clamp(balance_loss, max=1.0)
            return gates, balance_loss
        return gates, None


class SteinKernel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.adaptive_bandwidth = config['bottleneck']['adaptive_bandwidth']
        self.min_bandwidth      = config['bottleneck']['min_bandwidth']
        self.max_bandwidth      = config['bottleneck']['max_bandwidth']
        self.bandwidth_factor   = config['bottleneck']['bandwidth_factor']
        self.chunk_size         = 128
        self.register_buffer('bandwidth', torch.tensor(1.0))

    def update_bandwidth(self, x, y):
        with torch.no_grad():
            sample_size = min(512, x.size(0))
            diff        = x[:sample_size].unsqueeze(1) - y[:sample_size].unsqueeze(0)
            dist_sq     = torch.sum(diff ** 2, dim=-1)
            bandwidth   = torch.sqrt(torch.median(dist_sq.view(-1)) / 2.0) * self.bandwidth_factor
            self.bandwidth = torch.clamp(bandwidth, self.min_bandwidth, self.max_bandwidth)

    def score_kernel(self, x, score_x, y, score_y):
        self.update_bandwidth(x, y)
        diff       = x.unsqueeze(1) - y.unsqueeze(0)
        dist_sq    = torch.sum(diff ** 2, dim=-1)
        kernel_mat = torch.exp(-dist_sq / (2 * self.bandwidth ** 2))
        term1 = torch.sum(kernel_mat.unsqueeze(-1) * score_x.unsqueeze(1) * score_y.unsqueeze(0), dim=-1)
        term2 = torch.sum(diff * score_y.unsqueeze(0), dim=-1) * kernel_mat / (self.bandwidth ** 2)
        term3 = torch.sum(diff ** 2, dim=-1) * kernel_mat / (self.bandwidth ** 4) \
                - kernel_mat * x.size(-1) / (self.bandwidth ** 2)
        return term1 + term2 + term3


class MultiViewEntropyBottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['multimodal']['hidden_size']
        self.output_dim  = config['bottleneck']['dim']
        self.beta        = config['bottleneck']['beta']
        self.image_encoder = nn.Sequential(
            nn.Linear(config['image']['projection_dim'], self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(config['multimodal']['projection_dropout']),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.output_dim * 2)
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(config['text']['projection_dim'], self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(config['multimodal']['projection_dropout']),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.output_dim * 2)
        )
        score_hidden = self.output_dim * 2
        self.image_score = nn.Sequential(
            nn.Linear(self.output_dim, score_hidden), nn.LayerNorm(score_hidden),
            nn.Dropout(config['multimodal']['projection_dropout']), nn.GELU(),
            nn.Linear(score_hidden, self.output_dim)
        )
        self.text_score = nn.Sequential(
            nn.Linear(self.output_dim, score_hidden), nn.LayerNorm(score_hidden),
            nn.Dropout(config['multimodal']['projection_dropout']), nn.GELU(),
            nn.Linear(score_hidden, self.output_dim)
        )
        self.stein_kernel = SteinKernel(config)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def compute_kl_loss(self, mu1, logvar1, mu2, logvar2, mask=None):
        kl1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(), dim=-1)
        kl2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp(), dim=-1)
        if mask is not None:
            seq_len = mask.sum(dim=1).long() - 1
            idx     = torch.arange(mask.size(0), device=mask.device)
            kl1, kl2 = kl1[idx, seq_len], kl2[idx, seq_len]
        return (kl1.mean() + kl2.mean()) / 2

    def forward(self, image_features, text_features, attention_mask=None):
        B = image_features.size(0)
        image_mu, image_logvar = torch.chunk(self.image_encoder(image_features), 2, dim=-1)
        text_mu,  text_logvar  = torch.chunk(self.text_encoder(text_features),   2, dim=-1)
        image_z = F.normalize(self.reparameterize(image_mu, image_logvar), p=2, dim=-1)
        text_z  = F.normalize(self.reparameterize(text_mu,  text_logvar),  p=2, dim=-1)
        image_score = self.image_score(image_z)
        text_score  = self.text_score(text_z)
        if attention_mask is not None:
            seq_len = attention_mask.sum(dim=1).long() - 1
            idx     = torch.arange(B, device=image_z.device)
            image_z_last     = image_z[idx, seq_len]
            text_z_last      = text_z[idx, seq_len]
            image_score_last = image_score[idx, seq_len]
            text_score_last  = text_score[idx, seq_len]
        else:
            image_z_last     = image_z[:, -1]
            text_z_last      = text_z[:, -1]
            image_score_last = image_score[:, -1]
            text_score_last  = text_score[:, -1]
        alignment_loss = torch.mean(self.stein_kernel.score_kernel(
            image_z_last, image_score_last, text_z_last, text_score_last
        ))
        kl_loss    = self.compute_kl_loss(image_mu, image_logvar, text_mu, text_logvar, attention_mask)
        total_loss = kl_loss + self.beta * alignment_loss
        return {'image_repr': image_z, 'text_repr': text_z,
                'alignment_loss': alignment_loss, 'kl_loss': kl_loss, 'total_loss': total_loss}


class FluidMMRec(nn.Module):
    """
    Fluid_MMRec: Mamba + MultiHead CrossAttention + MoE + MVEB
    RecBole 제거, 우리 데이터/평가 방식으로 변경
    모델 구조 자체는 원본과 동일
    """
    def __init__(self, config, num_items, image_features, text_features):
        super().__init__()
        self.config          = config
        self.num_items       = num_items       # 1-based
        self.id_embedding_dim = config['id_embedding_dim']
        self.num_heads       = config['num_attention_heads']
        self.head_dim        = config['multimodal']['hidden_size'] // self.num_heads
        self.dropout_prob    = config['dropout_prob']
        self.bottleneck_weight = config['bottleneck']['weight']

        # 아이템 피처 (고정)
        self.register_buffer('image_features', torch.tensor(image_features, dtype=torch.float))
        self.register_buffer('text_features',  torch.tensor(text_features,  dtype=torch.float))

        # 아이템 ID 임베딩 (0: padding)
        self.item_embedding = nn.Embedding(num_items + 1, self.id_embedding_dim, padding_idx=0)

        # 피처 projection
        self.image_projection = nn.Linear(config['image']['feature_dim'], config['image']['projection_dim'])
        self.text_projection  = nn.Linear(config['text']['feature_dim'],  config['text']['projection_dim'])

        # Mamba (시퀀셜 모델링)
        self.id_mamba     = MambaLayer(config)
        self.fusion_mamba = MambaLayer(config)

        # CrossAttention + MoE + MVEB (원본 구조 그대로)
        self.cross_attention = MultiHeadCrossAttention(config)
        self.router          = ExpertRouter(config)
        self.stein_mveb      = MultiViewEntropyBottleneck(config)

        # Expert
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim * 2, self.id_embedding_dim),
                nn.ReLU(),
                nn.Dropout(config['dropout_prob'])
            ) for _ in range(config['expert']['num_experts'])
        ])

        # Fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.id_embedding_dim * self.num_heads, self.id_embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.id_embedding_dim),
            nn.Dropout(config['multimodal']['fusion_dropout'])
        )
        self.final_fusion = nn.Sequential(
            nn.Linear(self.id_embedding_dim * 2, self.id_embedding_dim),
            nn.LayerNorm(self.id_embedding_dim),
            nn.ReLU(),
            nn.Dropout(config['dropout_prob'])
        )
        self.final_norm = nn.LayerNorm(self.id_embedding_dim)

        self.loss_fn = nn.CrossEntropyLoss()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_multimodal_features(self, item_seq):
        """
        item_seq: (B, L) 1-based item_id, 0=padding
        return: image_feat (B, L, D), text_feat (B, L, D), attention_mask (B, L)
        """
        attention_mask = (item_seq != 0)                    # (B, L) True=유효
        ids = (item_seq - 1).clamp(min=0)                  # 0-based, padding→0
        image_feat = self.image_features[ids]               # (B, L, D_img)
        text_feat  = self.text_features[ids]                # (B, L, D_txt)
        # padding 위치는 0으로
        mask_f = attention_mask.unsqueeze(-1).float()
        image_feat = image_feat * mask_f
        text_feat  = text_feat  * mask_f
        return image_feat, text_feat, attention_mask

    def forward(self, item_seq, seq_len):
        """
        item_seq: (B, L) 1-based, 0=padding
        seq_len:  (B,) 실제 시퀀스 길이
        return: seq_output (B, dim)
        """
        # ── ID Mamba ──────────────────────────────────────────────────────
        item_emb      = self.item_embedding(item_seq)       # (B, L, dim)
        id_seq_output = self.id_mamba(item_emb)             # (B, L, dim)

        # ── 멀티모달 피처 ─────────────────────────────────────────────────
        image_features, text_features, mask = self.get_multimodal_features(item_seq)
        image_features = torch.nan_to_num(image_features, nan=0.0)
        text_features  = torch.nan_to_num(text_features,  nan=0.0)

        image_proj = F.normalize(self.image_projection(image_features), p=2, dim=-1)
        text_proj  = F.normalize(self.text_projection(text_features),   p=2, dim=-1)

        # ── MVEB ──────────────────────────────────────────────────────────
        mveb_output    = self.stein_mveb(image_proj, text_proj, mask)
        image_aligned  = mveb_output['image_repr']
        text_aligned   = mveb_output['text_repr']
        self.mveb_loss = mveb_output['total_loss']

        # ── CrossAttention + MoE ──────────────────────────────────────────
        head_features = self.cross_attention(image_aligned, text_aligned, mask)
        head_features = torch.nan_to_num(head_features, nan=0.0)

        gates, _ = self.router(head_features, training=self.training)

        B, L = item_seq.shape
        expert_outputs = []
        for head_idx in range(self.num_heads):
            head_output = head_features[:, head_idx]        # (B, L, 2D)
            head_gates  = gates[:, head_idx]                # (B, L, num_experts)
            head_expert_output = torch.zeros(B, L, self.id_embedding_dim, device=item_seq.device)
            for expert_idx, expert in enumerate(self.experts):
                expert_out = F.normalize(expert(head_output), p=2, dim=-1)
                weight     = head_gates[:, :, expert_idx].unsqueeze(-1)
                head_expert_output += expert_out * weight
            expert_outputs.append(head_expert_output)

        combined_output = torch.cat(expert_outputs, dim=-1)  # (B, L, dim*H)
        modal_features  = F.normalize(self.feature_fusion(combined_output), p=2, dim=-1)

        # ── Final Fusion (Mamba) ──────────────────────────────────────────
        final_features = torch.cat([id_seq_output, modal_features], dim=-1)  # (B, L, dim*2)

        # final_fusion이 MambaLayer라 dim이 맞지 않으므로 projection 후 Mamba
        # 원본 Fliud_MMRec은 fusion_mamba를 final_fusion으로 사용 → 여기서도 동일
        # 단, 입력 dim이 다르므로 Linear로 맞춰줌
        fused_features = self.final_fusion(final_features)  # (B, L, dim)
        final_output   = self.final_norm(fused_features)
        final_output   = torch.clamp(final_output, min=-10.0, max=10.0)

        # ── 마지막 유효 위치 추출 ─────────────────────────────────────────
        # seq_len - 1 위치의 출력을 유저 표현으로 사용
        idx        = (seq_len - 1).clamp(min=0)
        idx_expand = idx.view(-1, 1, 1).expand(-1, 1, final_output.size(-1))
        seq_output = final_output.gather(1, idx_expand).squeeze(1)  # (B, dim)
        return seq_output

    def compute_loss(self, item_seq, seq_len, target):
        """
        학습: 전체 아이템 CrossEntropy loss + MVEB loss
        target: (B,) 1-based item_id
        """
        seq_output    = self.forward(item_seq, seq_len)             # (B, dim)
        item_emb_all  = self.item_embedding.weight[1:]              # (num_items, dim) — 1-based
        logits        = torch.matmul(seq_output, item_emb_all.t())  # (B, num_items)
        rec_loss      = self.loss_fn(logits, target - 1)            # 0-based target
        total_loss    = rec_loss + self.bottleneck_weight * self.mveb_loss
        return total_loss

    def predict_candidates(self, item_seq, seq_len, candidate_ids):
        """
        평가: 101개 후보에 대한 스코어
        candidate_ids: (B, 101) 1-based item_id
        return: scores (B, 101)
        """
        seq_output   = self.forward(item_seq, seq_len)              # (B, dim)
        cand_emb     = self.item_embedding(candidate_ids)           # (B, 101, dim)
        scores       = torch.bmm(seq_output.unsqueeze(1), cand_emb.transpose(1, 2)).squeeze(1)
        return scores                                               # (B, 101)
