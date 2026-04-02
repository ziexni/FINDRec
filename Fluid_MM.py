import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm

from fluid_mmrec import MultiModalMoERec


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Dataset ───────────────────────────────────────────────────────────────────

class FluidDataset(Dataset):
    """
    leave-two-out 분할 기반 Dataset
    - item_id: 1-based (0=padding)
    - train: 마지막 2개 제외
    - valid: 뒤에서 2번째
    - test:  마지막 (시퀀스: train + valid)
    """
    def __init__(self, samples, max_seq_len):
        self.samples     = samples
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, seq, target = self.samples[idx]
        seq     = seq[-self.max_seq_len:]
        seq_len = len(seq)
        pad_len = self.max_seq_len - seq_len
        item_ids = seq + [0] * pad_len
        return {
            'user_id':  torch.tensor(u,        dtype=torch.long),
            'item_ids': torch.tensor(item_ids, dtype=torch.long),
            'seq_len':  torch.tensor(seq_len,  dtype=torch.long),
            'target':   torch.tensor(target,   dtype=torch.long),
        }


def load_data(interaction_path, item_path, title_npy_path, max_seq_len=50):
    """
    우리 데이터 로딩 + leave-two-out 분할
    - item_id: 1-based (0=padding)
    - train: 마지막 2개 제외 → 학습 샘플: input=train[:-1], target=train[-1]
    - valid: 뒤에서 2번째   → 평가 샘플: input=train, target=valid
    - test:  마지막          → 평가 샘플: input=train+valid, target=test
    """
    interactions_df = pd.read_parquet(interaction_path).sort_values(['user_id', 'timestamp'])
    item_df         = pd.read_parquet(item_path).reset_index(drop=True)
    title_feat      = np.load(title_npy_path)

    interactions_df = interactions_df.copy()
    interactions_df['item_id'] = interactions_df['item_id'] + 1  # 1-based
    num_items = int(interactions_df['item_id'].max())

    image_features = np.stack(item_df['video_feature'].values)  # (num_items, D_img)
    text_features  = title_feat                                  # (num_items, D_txt)

    user_sequences = defaultdict(list)
    for _, row in interactions_df.iterrows():
        user_sequences[int(row['user_id'])].append(int(row['item_id']))

    train_samples   = []
    valid_samples   = []
    test_samples    = []
    train_item_dict = defaultdict(set)

    for u, seq in user_sequences.items():
        if len(seq) < 3:
            continue
        train_seq  = seq[:-2]
        valid_item = seq[-2]
        test_item  = seq[-1]

        for s in train_seq:
            train_item_dict[u].add(s)

        # train 샘플
        if len(train_seq) >= 2:
            train_samples.append((u, train_seq[:-1], train_seq[-1]))

        # valid 샘플: input=train_seq, target=valid_item
        valid_samples.append((u, train_seq, valid_item))

        # test 샘플: input=train_seq+valid_item, target=test_item
        test_samples.append((u, train_seq + [valid_item], test_item))

    return (train_samples, valid_samples, test_samples,
            num_items, image_features, text_features, train_item_dict)


# ── 평가 함수 ─────────────────────────────────────────────────────────────────

def evaluate(model, data_loader, train_item_dict, num_items,
             topk=10, num_neg=100, device='cuda'):
    """
    101개 후보 기반 평가 (베이스라인과 동일)
    - 정답 1 + negative 100
    - negative: train 아이템 제외
    - NDCG@10, HR@10, MRR
    """
    model.eval()
    NDCG = HR = MRR = 0.0
    count = 0
    np.random.seed(42)

    for batch in data_loader:
        item_ids = batch['item_ids'].to(device)
        seq_len  = batch['seq_len'].to(device)
        targets  = batch['target']
        user_ids = batch['user_id'].tolist()
        B        = item_ids.size(0)

        candidates = []
        for b in range(B):
            target = targets[b].item()
            rated  = train_item_dict[user_ids[b]] | {target}
            cands  = [target]
            for _ in range(num_neg):
                neg = np.random.randint(1, num_items + 1)
                while neg in rated:
                    neg = np.random.randint(1, num_items + 1)
                cands.append(neg)
            candidates.append(cands)

        candidates = torch.tensor(candidates, dtype=torch.long, device=device)

        with torch.no_grad():
            scores = model.predict_candidates(item_ids, seq_len, candidates)

        ranks = (-scores).argsort(dim=1).argsort(dim=1)[:, 0]
        for rank in ranks.cpu().tolist():
            count += 1
            MRR   += 1 / (rank + 1)
            if rank < topk:
                NDCG += 1 / np.log2(rank + 2)
                HR   += 1

    N = max(count, 1)
    return NDCG / N, HR / N, MRR / N


# ── 학습 ─────────────────────────────────────────────────────────────────────

def train(config):
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading data...")
    (train_samples, valid_samples, test_samples,
     num_items, image_features, text_features,
     train_item_dict) = load_data(
        config['interaction_path'],
        config['item_path'],
        config['title_npy_path'],
        max_seq_len=config['max_seq_len']
    )
    print(f"Items: {num_items} | Train: {len(train_samples)} | "
          f"Valid: {len(valid_samples)} | Test: {len(test_samples)}")

    config['image']['feature_dim'] = image_features.shape[1]
    config['text']['feature_dim']  = text_features.shape[1]

    train_dataset = FluidDataset(train_samples, config['max_seq_len'])
    valid_dataset = FluidDataset(valid_samples, config['max_seq_len'])
    test_dataset  = FluidDataset(test_samples,  config['max_seq_len'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True,  num_workers=config['num_workers'])
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'],
                              shuffle=False, num_workers=config['num_workers'])
    test_loader  = DataLoader(test_dataset,  batch_size=config['batch_size'],
                              shuffle=False, num_workers=config['num_workers'])

    model = MultiModalMoERec(config, num_items, image_features, text_features).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=config['learning_rate'],
                                   weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6
    )

    best_val_ndcg  = 0.0
    best_test_ndcg = 0.0
    best_test_hr   = 0.0
    best_test_mrr  = 0.0
    num_decreases  = 0

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            item_ids = batch['item_ids'].to(device)
            seq_len  = batch['seq_len'].to(device)
            target   = batch['target'].to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(item_ids, seq_len, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {avg_loss:.4f}")

        if (epoch + 1) % config['eval_step'] == 0:
            # valid: train 시퀀스 기준
            val_ndcg, val_hr, val_mrr = evaluate(
                model, valid_loader, train_item_dict, num_items,
                topk=config['topk'], num_neg=config['num_neg'], device=device
            )
            print(f"  [Valid] NDCG@{config['topk']}: {val_ndcg:.4f} | "
                  f"HR@{config['topk']}: {val_hr:.4f} | MRR: {val_mrr:.4f}")

            # test: train+valid 시퀀스 기준
            test_ndcg, test_hr, test_mrr = evaluate(
                model, test_loader, train_item_dict, num_items,
                topk=config['topk'], num_neg=config['num_neg'], device=device
            )
            print(f"  [Test]  NDCG@{config['topk']}: {test_ndcg:.4f} | "
                  f"HR@{config['topk']}: {test_hr:.4f} | MRR: {test_mrr:.4f}")

            if val_ndcg > best_val_ndcg:
                best_val_ndcg  = val_ndcg
                best_test_ndcg = test_ndcg
                best_test_hr   = test_hr
                best_test_mrr  = test_mrr
                num_decreases  = 0
                torch.save(model.state_dict(), config['save_path'])
                print(f"  ✓ Best model saved (Val NDCG: {best_val_ndcg:.4f})")
            else:
                num_decreases += 1
                if num_decreases >= config['stopping_step']:
                    print("Early stopping.")
                    break

    print(f"\n{'='*50}")
    print(f"Best Valid NDCG@{config['topk']}: {best_val_ndcg:.4f}")
    print(f"Best Test  NDCG@{config['topk']}: {best_test_ndcg:.4f} | "
          f"HR@{config['topk']}: {best_test_hr:.4f} | MRR: {best_test_mrr:.4f}")
    print(f"{'='*50}")
