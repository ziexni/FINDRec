import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from vlgraph_data_utils import data_partition, build_item_modality, Data
from model import VLGraph, train_and_test  # 원본 model.py 그대로 사용


def build_data_from_sequences(user_seqs, user_valid_or_test,
                               item_to_image, item_to_title,
                               mode='train'):
    """
    user_train / user_valid / user_test 딕셔너리로부터
    원본 Data 클래스 입력 형식 (inputs, image_inputs, text_inputs, targets) 구성

    mode='train': input=train[:-1], target=train[-1]
    mode='valid': input=train,      target=valid[0]
    mode='test' : input=train+valid,target=test[0]   ← 베이스라인 프로토콜
    """
    inputs       = []
    image_inputs = []
    text_inputs  = []
    targets      = []

    for u, seq in user_seqs.items():
        gt = user_valid_or_test.get(u, [])
        if not gt:
            continue

        if mode == 'train':
            if len(seq) < 2:
                continue
            input_seq = seq[:-1]
            target    = seq[-1]
        else:
            input_seq = seq  # valid: train / test: train+valid (호출 시 이미 합쳐서 전달)
            target    = gt[0]

        # 아이템 시퀀스 → image/title 노드 리스트 변환
        img_seq = [item_to_image.get(item, [0]) for item in input_seq]
        txt_seq = [item_to_title.get(item, [0]) for item in input_seq]

        inputs.append(list(input_seq))
        image_inputs.append(img_seq)
        text_inputs.append(txt_seq)
        targets.append(target)

    return inputs, image_inputs, text_inputs, targets


def evaluate(model, dataset, eval_dict, train_item_dict, itemnum,
             mode='valid', num_neg=100, topk=10, batch_size=64):
    """
    101개 후보 기반 평가 (정답 1 + negative 100)
    베이스라인 프로토콜:
    - valid: 유저 수 > 1000  → 1000명 랜덤 샘플링
    - test : 유저 수 > 10000 → 10000명 랜덤 샘플링
    """
    import copy

    all_valid_users = [u for u, items in eval_dict.items() if len(items) > 0]

    sample_limit = 1000 if mode == 'valid' else 10000
    if len(all_valid_users) > sample_limit:
        rng = np.random.RandomState(42)
        valid_users = rng.choice(all_valid_users, size=sample_limit, replace=False).tolist()
    else:
        valid_users = all_valid_users

    NDCG = HR = MRR = 0.0
    count = 0
    np.random.seed(42)

    eval_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2)

    # 유저별 예측 점수 수집
    # dataset의 순서가 valid_users와 같아야 하므로
    # dataset은 valid_users 순서로 구성되어 있다고 가정
    model.eval()
    all_scores = []
    all_targets = []

    for data in eval_loader:
        adj, nodes, node_type_mask, node_pos_matrix, inputs_mask, targets, \
            u_input, alias_inputs, alias_img_inputs, alias_txt_inputs = data

        adj             = adj.float().cuda()
        node_pos_matrix = node_pos_matrix.float().cuda()
        nodes           = nodes.long().cuda()
        node_type_mask  = node_type_mask.long().cuda()
        alias_inputs    = alias_inputs.long().cuda()
        alias_img_inputs = alias_img_inputs.long().cuda()
        alias_txt_inputs = alias_txt_inputs.long().cuda()
        inputs_mask     = inputs_mask.long().cuda()
        targets         = targets.long().cuda()

        with torch.no_grad():
            node_hidden = model.forward(adj, nodes, node_type_mask,
                                        node_pos_matrix, stage='test')
            scores = model.compute_full_scores(node_hidden, alias_inputs,
                                               alias_img_inputs, alias_txt_inputs,
                                               inputs_mask)
        all_scores.append(scores.cpu())
        all_targets.append(targets.cpu())

    all_scores  = torch.cat(all_scores,  dim=0)  # (N, itemnum)
    all_targets = torch.cat(all_targets, dim=0)  # (N,)

    for idx, u in enumerate(valid_users):
        if idx >= len(all_scores):
            break
        target = all_targets[idx].item() - 1   # 0-based for scores
        rated  = train_item_dict.get(u, set())

        # 101개 후보 구성 (정답 1 + negative 100)
        cands = [target]
        for _ in range(num_neg):
            neg = np.random.randint(0, itemnum)
            while (neg + 1) in rated or neg == target:
                neg = np.random.randint(0, itemnum)
            cands.append(neg)

        cand_scores = all_scores[idx][cands]
        rank = (-cand_scores).argsort().argsort()[0].item()

        count += 1
        MRR   += 1 / (rank + 1)
        if rank < topk:
            NDCG += 1 / np.log2(rank + 2)
            HR   += 1

    N = max(count, 1)
    return NDCG / N, HR / N, MRR / N


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 데이터 로드 ───────────────────────────────────────────────────────
    print("Loading data...")
    user_train, user_valid, user_test, usernum, itemnum = \
        data_partition(config['interaction_path'])

    item_df    = pd.read_parquet(config['item_path'])
    title_feat = np.load(config['title_npy_path'])
    image_feat = np.stack(item_df['video_feature'].values)

    # 아이템 → image/title 노드 ID 매핑
    item_to_image, item_to_title = build_item_modality(
        config['item_path'], title_feat, itemnum
    )

    # train_item_dict: 유저별 학습 아이템 집합 (negative 샘플링용)
    train_item_dict = {u: set(seq) for u, seq in user_train.items()}

    # ── 학습 데이터 구성 ──────────────────────────────────────────────────
    train_inputs, train_img, train_txt, train_targets = build_data_from_sequences(
        user_train, user_train, item_to_image, item_to_title, mode='train'
    )
    train_data = Data(
        (train_inputs, train_img, train_txt, train_targets),
        item_to_image, item_to_title, link_k=config['link_k']
    )

    # ── valid 데이터 구성 (컨텍스트: train) ──────────────────────────────
    valid_inputs, valid_img, valid_txt, valid_targets = build_data_from_sequences(
        user_train, user_valid, item_to_image, item_to_title, mode='valid'
    )
    valid_data = Data(
        (valid_inputs, valid_img, valid_txt, valid_targets),
        item_to_image, item_to_title,
        link_k=config['link_k'], train_len=train_data.max_len
    )

    # ── test 데이터 구성 (컨텍스트: train + valid) ────────────────────────
    # 베이스라인 프로토콜: test 시 valid 아이템까지 컨텍스트에 포함
    user_train_plus_valid = {}
    for u in user_train:
        extra = [user_valid[u][0]] if user_valid.get(u) else []
        user_train_plus_valid[u] = user_train[u] + extra

    test_inputs, test_img, test_txt, test_targets = build_data_from_sequences(
        user_train_plus_valid, user_test, item_to_image, item_to_title, mode='valid'
    )
    test_data = Data(
        (test_inputs, test_img, test_txt, test_targets),
        item_to_image, item_to_title,
        link_k=config['link_k'], train_len=train_data.max_len
    )

    # ── image/title 클러스터 피처 (원본 VLGraph 형식에 맞게) ─────────────
    # 원본은 image_cluster_feature, text_cluster_feature를 받음
    # 우리는 아이템별 피처를 그대로 사용 (클러스터링 없이)
    # node ID offset에 맞게 인덱스 재구성
    # image 노드 ID = itemnum + item_id (1-based)
    # → image_cluster_feature[img_node_id - itemnum - 1] = image_feat[item_id - 1]
    image_cluster_feature = image_feat  # (num_items, img_dim)
    text_cluster_feature  = title_feat  # (num_items, title_dim)

    # item_image_list / item_text_list: [num_items, K] 형태 (원본 형식)
    # K=1이므로 각 아이템의 image/title 노드 ID 1개씩
    num_items_raw = len(item_df)
    item_image_list = [[itemnum + i + 1] for i in range(num_items_raw)]  # 1-based
    item_text_list  = [[itemnum * 2 + i + 1] for i in range(num_items_raw)]

    config['num_node'] = {config['dataset']: itemnum * 3 + 1}  # 원본 형식
    config['cluster_num'] = {config['dataset']: num_items_raw}

    # ── 모델 초기화 (원본 VLGraph 그대로) ────────────────────────────────
    model = VLGraph(config, image_cluster_feature, text_cluster_feature,
                    item_image_list, item_text_list).cuda()

    train_loader = DataLoader(train_data, batch_size=config['batch_size'],
                              shuffle=True, num_workers=config['num_workers'],
                              pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=config['eval_batch_size'],
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_data,  batch_size=config['eval_batch_size'],
                              shuffle=False, num_workers=2, pin_memory=True)

    best_val_ndcg  = 0.0
    best_test_ndcg = 0.0
    best_test_hr   = 0.0
    best_test_mrr  = 0.0
    num_decreases  = 0

    for epoch in range(config['num_epochs']):
        # ── 학습 (원본 train_and_test의 train 부분) ───────────────────────
        model.train()
        total_loss = 0.0
        for data in train_loader:
            model.optimizer.zero_grad()
            adj, nodes, node_type_mask, node_pos_matrix, inputs_mask, targets, \
                u_input, alias_inputs, alias_img_inputs, alias_txt_inputs = data

            adj             = adj.float().cuda()
            node_pos_matrix = node_pos_matrix.float().cuda()
            nodes, node_type_mask = nodes.long().cuda(), node_type_mask.long().cuda()
            alias_inputs    = alias_inputs.long().cuda()
            alias_img_inputs = alias_img_inputs.long().cuda()
            alias_txt_inputs = alias_txt_inputs.long().cuda()
            inputs_mask     = inputs_mask.long().cuda()
            targets         = targets.long().cuda()

            node_hidden = model.forward(adj, nodes, node_type_mask,
                                        node_pos_matrix, stage='train')
            scores = model.compute_full_scores(node_hidden, alias_inputs,
                                              alias_img_inputs, alias_txt_inputs,
                                              inputs_mask)
            loss = model.loss_function(scores, targets - 1)
            loss.backward()
            model.optimizer.step()
            total_loss += loss.item()

        model.scheduler.step()
        print(f"Epoch {epoch+1}/{config['num_epochs']} | Loss: {total_loss/len(train_loader):.4f}")

        # ── 주기적 평가 ───────────────────────────────────────────────────
        if (epoch + 1) % config['eval_every'] == 0:
            val_ndcg, val_hr, val_mrr = evaluate(
                model, valid_data, user_valid, train_item_dict, itemnum,
                mode='valid', num_neg=config['num_neg'], topk=config['topk'],
                batch_size=config['eval_batch_size']
            )
            print(f"  [Valid] NDCG@{config['topk']}: {val_ndcg:.4f} | "
                  f"HR@{config['topk']}: {val_hr:.4f} | "
                  f"MRR@{config['topk']}: {val_mrr:.4f}")

            test_ndcg, test_hr, test_mrr = evaluate(
                model, test_data, user_test, train_item_dict, itemnum,
                mode='test', num_neg=config['num_neg'], topk=config['topk'],
                batch_size=config['eval_batch_size']
            )
            print(f"  [Test]  NDCG@{config['topk']}: {test_ndcg:.4f} | "
                  f"HR@{config['topk']}: {test_hr:.4f} | "
                  f"MRR@{config['topk']}: {test_mrr:.4f}")

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
                if num_decreases >= config['patience']:
                    print("Early stopping.")
                    break

    print(f"\n{'='*50}")
    print(f"Best Valid NDCG@{config['topk']}: {best_val_ndcg:.4f}")
    print(f"Best Test  NDCG@{config['topk']}: {best_test_ndcg:.4f} | "
          f"HR@{config['topk']}: {best_test_hr:.4f} | "
          f"MRR: {best_test_mrr:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    config = {
        # 데이터 경로
        'interaction_path': './interaction.parquet',
        'item_path':        './item_used.parquet',
        'title_npy_path':   './title_emb.npy',
        'save_path':        './vlgraph_best.pt',
        'dataset':          'microvideo',

        # 원본 VLGraph 하이퍼파라미터
        'embedding_size':  128,
        'n_layer':           2,
        'aggregator':    'rgat',
        'max_relid':        10,
        'alpha':           0.2,
        'link_k':            1,   # 아이템당 image/title 노드 수 (K=1)
        'fusion_type':   'gate',
        'dropout_local':   0.2,
        'dropout_atten':   0.2,
        'seq_pooling':  'attention',
        'modality_prediction': True,
        'auxiliary_info': ['node_type', 'pos'],

        # 학습 설정
        'lr':             1e-3,
        'l2':             1e-5,
        'lr_dc':           0.1,
        'lr_dc_step':        3,
        'batch_size':      128,
        'num_workers':       3,
        'num_epochs':      100,
        'eval_every':        5,
        'topk':             10,
        'patience':          3,
        'num_neg':         100,
        'eval_batch_size':  64,
    }

    train(config)
