import sys
import warnings
import torch
warnings.filterwarnings('ignore')

from main_fluid import train

if __name__ == '__main__':
    config = {
        # ── 데이터 경로 ───────────────────────────────────────────────────
        'interaction_path': './interaction.parquet',
        'item_path':        './item_used.parquet',
        'title_npy_path':   './title_emb.npy',
        'save_path':        './best_fluid.pt',

        # ── 모델 구조 (원본 config.yaml과 동일) ───────────────────────────
        'id_embedding_dim':     128,
        'num_attention_heads':    8,
        'dropout_prob':         0.2,
        'bottleneck': {
            'dim':                256,
            'beta':               1.0,
            'weight':             0.1,
            'kernel_type':       'rbf',
            'bandwidth_factor':   1.0,
            'adaptive_bandwidth': True,
            'min_bandwidth':      0.1,
            'max_bandwidth':     10.0,
        },
        'mamba': {
            'd_state':    16,
            'd_conv':      4,
            'expand':      2,
            'norm_eps':  1e-5,
            'hidden_dim': 128,
            'num_layers':   2,
        },
        'multimodal': {
            'hidden_size':        256,
            'projection_dropout': 0.2,
            'fusion_dropout':     0.2,
        },
        'expert': {
            'num_experts': 4,
        },
        'router': {
            'hidden_size': 256,
            'dropout':     0.1,
        },
        'image': {
            'feature_dim':    None,  # load_data에서 자동 설정
            'projection_dim': 256,
        },
        'text': {
            'feature_dim':    None,  # load_data에서 자동 설정
            'projection_dim': 256,
        },

        # ── 학습 파라미터 (원본 config.yaml과 동일) ───────────────────────
        'seed':                42,
        'epochs':             100,
        'batch_size':         128,
        'num_workers':          3,
        'learning_rate':     0.001,
        'weight_decay':       0.01,
        'gradient_clip_norm':  2.0,
        'eval_step':            1,
        'stopping_step':        5,
        'max_seq_len':         50,

        # ── 평가 (베이스라인과 동일) ──────────────────────────────────────
        'topk':    10,   # NDCG@10, HR@10, MRR
        'num_neg': 100,  # 101개 후보 (정답 1 + negative 100)
    }

    train(config)
