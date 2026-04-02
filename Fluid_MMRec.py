import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import numpy as np


# ── 원본 구조 그대로 유지 ─────────────────────────────────────────────────────

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


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
        self.num_heads   = config['num_attention_heads']
        self.hidden_size = config['multimodal']['hidden_size']
        self.head_dim    = self.hidden_size // self.num_heads
        self.layer_norm  = nn.LayerNorm(self.hidden_size)
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

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, image_features, text_features, attention_mask=None):
        image_proj = self.image_projection(image_features)
        text_proj  = self.text_projection(text_features)
        image_proj = F.normalize(image_proj, p=2, dim=-1)
        text_proj  = F.normalize(text_proj,  p=2, dim=-1)
        image_heads = self.split_heads(image_proj)
        text_heads  = self.split_heads(text_proj)
        # 원본: attention_mask를 반전해서 key_padding_mask로 사용
        if attention_mask is not None:
            attention_mask = ~attention_mask
        head_outputs = []
        for head_idx in range(self.num_heads):
            curr_image = image_heads[:, head_idx] * self.scaling
            curr_text  = text_heads[:, head_idx]  * self.scaling
            img2text, _ = self.image_to_text_heads[head_idx](
                query=curr_image, key=curr_text, value=curr_text,
                key_padding_mask=attention_mask
            )
            text2img, _ = self.text_to_image_heads[head_idx](
                query=curr_text, key=curr_image, value=curr_image,
                key_padding_mask=attention_mask
            )
            img2text   = curr_image + img2text
            text2img   = curr_text  + text2img
            combined   = torch.cat([img2text, text2img], dim=-1)
            normalized = self.head_norms[head_idx](combined)
            head_outputs.append(normalized)
        return torch.stack(head_outputs, dim=1)  # (B, H, L, head_dim*2)


class ExpertRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config['expert']['num_experts']
        self.hidden_size = config['router']['hidden_size']
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
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
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
        self.kernel_type        = config['bottleneck']['kernel_type']
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
            bandwidth_sq = torch.median(dist_sq.view(-1))
            bandwidth    = torch.sqrt(bandwidth_sq / 2.0) * self.bandwidth_factor
            self.bandwidth = torch.clamp(bandwidth, min=self.min_bandwidth, max=self.max_bandwidth)

    def score_kernel(self, x, score_x, y, score_y):
        self.update_bandwidth(x, y)
        diff       = x.unsqueeze(1) - y.unsqueeze(0)
        dist_sq    = torch.sum(diff ** 2, dim=-1)
        kernel_mat = torch.exp(-dist_sq / (2 * self.bandwidth ** 2))
        term1 = kernel_mat.unsqueeze(-1) * score_x.unsqueeze(1)
        term1 = torch.sum(term1 * score_y.unsqueeze(0), dim=-1)
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
            nn.Linear(self.output_dim, score_hidden),
            nn.LayerNorm(score_hidden),
            nn.Dropout(config['multimodal']['projection_dropout']),
            nn.GELU(),
            nn.Linear(score_hidden, self.output_dim)
        )
        self.text_score = nn.Sequential(
            nn.Linear(self.output_dim, score_hidden),
            nn.LayerNorm(score_hidden),
            nn.Dropout(config['multimodal']['projection_dropout']),
            nn.GELU(),
            nn.Linear(score_hidden, self.output_dim)
        )
        self.stein_kernel = SteinKernel(config)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_kl_loss(self, mu1, logvar1, mu2, logvar2, mask=None):
        kl1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(), dim=-1)
        kl2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp(), dim=-1)
        if mask is not None:
            sequence_lengths = mask.sum(dim=1).long() - 1
            batch_indices    = torch.arange(mask.size(0), device=mask.device)
            kl1 = kl1[batch_indices, sequence_lengths]
            kl2 = kl2[batch_indices, sequence_lengths]
        return (kl1.mean() + kl2.mean()) / 2

    def forward(self, image_features, text_features, attention_mask=None):
        batch_size = image_features.size(0)
        image_params = self.image_encoder(image_features)
        text_params  = self.text_encoder(text_features)
        image_mu, image_logvar = torch.chunk(image_params, 2, dim=-1)
        text_mu,  text_logvar  = torch.chunk(text_params,  2, dim=-1)
        image_z = F.normalize(self.reparameterize(image_mu, image_logvar), p=2, dim=-1)
        text_z  = F.normalize(self.reparameterize(text_mu,  text_logvar),  p=2, dim=-1)
        image_score = self.image_score(image_z)
        text_score  = self.text_score(text_z)
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1).long() - 1
            batch_indices    = torch.arange(batch_size, device=image_z.device)
            image_z_last     = image_z[batch_indices, sequence_lengths]
            text_z_last      = text_z[batch_indices, sequence_lengths]
            image_score_last = image_score[batch_indices, sequence_lengths]
            text_score_last  = text_score[batch_indices, sequence_lengths]
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
        return {
            'image_repr': image_z, 'text_repr': text_z,
            'alignment_loss': alignment_loss, 'kl_loss': kl_loss, 'total_loss': total_loss
        }


class MultiModalMoERec(nn.Module):
    """
    원본 MultiModalMoERec 구조 그대로 유지
    RecBole(SequentialRecommender, BPRLoss) 제거
    우리 데이터/평가 방식으로 변경:
      - get_multimodal_features: RecBole dataset API → buffer 방식
      - gather_indexes → seq_len 기반 직접 추출
      - compute_loss / predict_candidates 추가
    """
    def __init__(self, config, num_items, image_features, text_features):
        super().__init__()
        self.config           = config
        self.num_items        = num_items
        self.id_embedding_dim = config['id_embedding_dim']
        self.modal_hidden_size = config['multimodal']['hidden_size']
        self.dropout_prob     = config['dropout_prob']
        self.num_heads        = config['num_attention_heads']
        self.head_dim         = self.modal_hidden_size // self.num_heads
        self.bottleneck_weight = config['bottleneck']['weight']

        # 아이템 피처 버퍼 (고정, 0-based index)
        self.register_buffer('image_features', torch.tensor(image_features, dtype=torch.float))
        self.register_buffer('text_features',  torch.tensor(text_features,  dtype=torch.float))

        # 원본과 동일한 레이어 구성
        self.item_embedding = nn.Embedding(num_items + 1, self.id_embedding_dim, padding_idx=0)

        self.image_projection = nn.Linear(config['image']['feature_dim'],
                                          config['image']['projection_dim'])
        self.text_projection  = nn.Linear(config['text']['feature_dim'],
                                          config['text']['projection_dim'])

        self.id_mamba     = MambaLayer(config)
        self.fusion_mamba = MambaLayer(config)

        self.cross_attention = MultiHeadCrossAttention(config)
        self.router          = ExpertRouter(config)
        self.stein_mveb      = MultiViewEntropyBottleneck(config)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim * 2, self.id_embedding_dim),
                nn.ReLU(),
                nn.Dropout(config['dropout_prob'])
            ) for _ in range(config['expert']['num_experts'])
        ])

        self.feature_fusion = nn.Sequential(
            nn.Linear(self.id_embedding_dim * config['num_attention_heads'],
                      self.id_embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.id_embedding_dim),
            nn.Dropout(config['multimodal']['fusion_dropout'])
        )

        # 원본 MultiModalMoERec의 final_fusion (Linear 방식)
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
        원본의 RecBole dataset.id2token() 방식 →
        우리 방식: 1-based item_id를 0-based index로 변환해서 buffer에서 조회
        """
        attention_mask = (item_seq != 0)                    # (B, L) True=유효
        ids = (item_seq - 1).clamp(min=0)                  # 0-based (padding→0)
        image_feat = self.image_features[ids]               # (B, L, D_img)
        text_feat  = self.text_features[ids]                # (B, L, D_txt)
        mask_f = attention_mask.unsqueeze(-1).float()
        image_feat = image_feat * mask_f                    # padding 위치 0
        text_feat  = text_feat  * mask_f
        return image_feat, text_feat, attention_mask

    def forward(self, item_seq, item_seq_len):
        """
        원본 forward와 동일한 흐름
        gather_indexes(RecBole 메서드) → seq_len 기반 직접 추출로 변경
        """
        # ── ID Mamba ──────────────────────────────────────────────────────
        item_emb      = self.item_embedding(item_seq)
        id_seq_output = self.id_mamba(item_emb)

        # ── 멀티모달 피처 ─────────────────────────────────────────────────
        image_features, text_features, mask = self.get_multimodal_features(item_seq)
        image_features = torch.nan_to_num(image_features, nan=0.0, posinf=1.0, neginf=-1.0)
        text_features  = torch.nan_to_num(text_features,  nan=0.0, posinf=1.0, neginf=-1.0)

        image_proj = self.image_projection(image_features)
        text_proj  = self.text_projection(text_features)
        image_proj = F.normalize(image_proj, p=2, dim=-1)
        text_proj  = F.normalize(text_proj,  p=2, dim=-1)

        # ── MVEB ──────────────────────────────────────────────────────────
        mveb_output    = self.stein_mveb(image_proj, text_proj, mask)
        image_aligned  = mveb_output['image_repr']
        text_aligned   = mveb_output['text_repr']
        self.mveb_loss = mveb_output['total_loss']

        # ── CrossAttention + MoE ──────────────────────────────────────────
        head_features = self.cross_attention(image_aligned, text_aligned, mask)
        head_features = torch.nan_to_num(head_features, nan=0.0, posinf=1.0, neginf=-1.0)

        gates, _ = self.router(head_features)

        batch_size = item_seq.size(0)
        seq_len    = item_seq.size(1)
        expert_outputs = []
        for head_idx in range(self.num_heads):
            head_output = head_features[:, head_idx]
            head_gates  = gates[:, head_idx]
            head_expert_output = torch.zeros(
                batch_size, seq_len, self.id_embedding_dim, device=item_seq.device
            )
            for expert_idx, expert in enumerate(self.experts):
                expert_out = expert(head_output)
                expert_out = F.normalize(expert_out, p=2, dim=-1)
                weight     = head_gates[:, :, expert_idx].unsqueeze(-1)
                head_expert_output += expert_out * weight
            expert_outputs.append(head_expert_output)

        combined_output = torch.cat(expert_outputs, dim=-1)
        modal_features  = self.feature_fusion(combined_output)
        modal_features  = F.normalize(modal_features, p=2, dim=-1)

        # ── Final Fusion ──────────────────────────────────────────────────
        final_features = torch.cat([id_seq_output, modal_features], dim=-1)
        fused_features = self.final_fusion(final_features)
        final_output   = self.final_norm(fused_features)
        final_output   = torch.clamp(final_output, min=-10.0, max=10.0)

        # 원본 gather_indexes(final_output, item_seq_len - 1) 대체
        idx        = (item_seq_len - 1).clamp(min=0)
        idx_expand = idx.view(-1, 1, 1).expand(-1, 1, final_output.size(-1))
        seq_output = final_output.gather(1, idx_expand).squeeze(1)  # (B, dim)
        return seq_output

    def compute_loss(self, item_seq, seq_len, target):
        """
        원본 calculate_loss (CE loss) + MVEB loss
        target: (B,) 1-based item_id
        """
        seq_output   = self.forward(item_seq, seq_len)
        test_item_emb = self.item_embedding.weight          # (num_items+1, dim)
        logits       = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        rec_loss     = self.loss_fn(logits, target)         # 원본은 pos_items 그대로 사용
        total_loss   = rec_loss + self.bottleneck_weight * self.mveb_loss
        return total_loss

    def predict_candidates(self, item_seq, seq_len, candidate_ids):
        """
        101개 후보에 대한 스코어 (평가용)
        candidate_ids: (B, 101) 1-based item_id
        """
        seq_output = self.forward(item_seq, seq_len)        # (B, dim)
        cand_emb   = self.item_embedding(candidate_ids)     # (B, 101, dim)
        scores     = torch.bmm(
            seq_output.unsqueeze(1), cand_emb.transpose(1, 2)
        ).squeeze(1)                                        # (B, 101)
        return scores
