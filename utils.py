import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class MicroVideoVLDataset(Dataset):
    """
    VLGraph용 Dataset — 우리 데이터 형식에 맞게 재작성
    - 세션 = user_train 시퀀스 (최대 max_seq_len개)
    - 노드 타입: item(1) / image(2) / title(3)
    - 엣지 타입:
        1: self-loop
        2: out (item_i → item_i+1, 동일 모달리티 내 순방향)
        3: in  (item_i+1 → item_i, 동일 모달리티 내 역방향)
        4: bi-direction (양방향 동시 등장)
        5: item → image  (inter-modality)
        6: image → item  (inter-modality)
        7: item → title  (inter-modality)
        8: title → item  (inter-modality)
        9: image → title (inter-modality)
       10: title → image (inter-modality)
    - leave-two-out: train / valid / test 분리
    """
    def __init__(self, interactions_df, image_feat, title_feat,
                 max_seq_len=50, mode='train'):
        self.max_seq_len = max_seq_len
        self.mode = mode
        # max_len: 노드 행렬 크기 (item + image + title 각 max_seq_len개)
        self.max_node_len = max_seq_len * 3

        self.user_train      = {}
        self.user_valid      = {}
        self.user_test       = {}
        self.train_item_dict = defaultdict(set)

        interactions_df = interactions_df.sort_values(['user_id', 'timestamp'])

        # ── 유저별 시퀀스 구성 ────────────────────────────────────────────
        user_sequences = defaultdict(list)
        for _, row in interactions_df.iterrows():
            u = int(row['user_id'])
            i = int(row['item_id']) + 1  # 0은 padding 전용 → 1-based
            user_sequences[u].append({
                'item_id':   i,
                'timestamp': float(row['timestamp']),
            })

        # num_items: 실제 아이템 ID 최댓값 + 1 (= 1-based max)
        # image/title 노드 offset에도 사용
        self.num_items = interactions_df['item_id'].max() + 2

        # ── leave-two-out 분리 ────────────────────────────────────────────
        for u, seq in user_sequences.items():
            if len(seq) < 3:
                self.user_train[u] = seq
                self.user_valid[u] = []
                self.user_test[u]  = []
            else:
                self.user_train[u] = seq[:-2]
                self.user_valid[u] = [seq[-2]]
                self.user_test[u]  = [seq[-1]]

            for s in self.user_train[u]:
                self.train_item_dict[u].add(s['item_id'])

        # ── 아이템 피처 (1-based, 인덱스 0은 zero padding) ───────────────
        # image_feat / title_feat: (num_items_raw, feat_dim), 0-based
        # → 앞에 zero row 추가해서 1-based 인덱싱
        pad_img   = np.zeros((1, image_feat.shape[1]), dtype=np.float32)
        pad_title = np.zeros((1, title_feat.shape[1]), dtype=np.float32)
        self.image_feat = np.concatenate([pad_img,   image_feat], axis=0)
        self.title_feat = np.concatenate([pad_title, title_feat], axis=0)

        # ── 학습 샘플 구성 ────────────────────────────────────────────────
        self.samples = []
        for u, train_seq in self.user_train.items():
            if mode == 'train':
                if len(train_seq) < 2:
                    continue
                target    = train_seq[-1]['item_id']
                input_seq = train_seq[:-1]
            elif mode == 'valid':
                if not self.user_valid[u]:
                    continue
                target    = self.user_valid[u][0]['item_id']
                input_seq = train_seq
            else:  # test: train + valid를 컨텍스트로 (베이스라인 프로토콜)
                if not self.user_test[u]:
                    continue
                target    = self.user_test[u][0]['item_id']
                valid_seq = self.user_valid[u] if self.user_valid[u] else []
                input_seq = train_seq + valid_seq

            self.samples.append((u, input_seq, target))

    def __len__(self):
        return len(self.samples)

    def _build_graph(self, item_ids):
        """
        아이템 ID 시퀀스로부터 이종 그래프 인접 행렬 및 노드 정보 구성
        - item 노드: item_id (1-based)
        - image 노드: item_id + num_items (offset)
        - title 노드: item_id + 2*num_items (offset)
        """
        le = len(item_ids)

        # 중복 제거 (순서 유지) — 같은 아이템이 여러 번 등장해도 노드는 1개
        u_nodes = list(dict.fromkeys(item_ids))
        i_nodes = [x + self.num_items     for x in u_nodes]  # image 노드
        t_nodes = [x + 2 * self.num_items for x in u_nodes]  # title 노드
        nodes_list = u_nodes + i_nodes + t_nodes
        node_num   = len(nodes_list)

        # 노드 벡터 (padding 포함)
        nodes = np.array(nodes_list + [0] * (self.max_node_len - node_num), dtype=np.int64)

        # 노드 타입 마스크: item=1, image=2, title=3, pad=0
        node_type_mask = (
            [1] * len(u_nodes) +
            [2] * len(i_nodes) +
            [3] * len(t_nodes) +
            [0] * (self.max_node_len - node_num)
        )

        # 노드 ID → 행렬 인덱스 매핑
        node2idx = {n: idx for idx, n in enumerate(nodes_list)}

        # ── 인접 행렬 구성 ────────────────────────────────────────────────
        adj = np.zeros((self.max_node_len, self.max_node_len), dtype=np.int32)

        # inter-modality 엣지: item ↔ image, item ↔ title, image ↔ title
        for item in u_nodes:
            item_idx = node2idx[item]
            img_idx  = node2idx[item + self.num_items]
            txt_idx  = node2idx[item + 2 * self.num_items]

            # self-loop
            adj[item_idx][item_idx] = 1
            adj[img_idx][img_idx]   = 1
            adj[txt_idx][txt_idx]   = 1

            # item ↔ image
            adj[item_idx][img_idx] = 5
            adj[img_idx][item_idx] = 6

            # item ↔ title
            adj[item_idx][txt_idx] = 7
            adj[txt_idx][item_idx] = 8

            # image ↔ title
            adj[img_idx][txt_idx] = 9
            adj[txt_idx][img_idx] = 10

        # intra-modality 순차 엣지: 연속 아이템 쌍 (item/image/title 각각)
        for pos in range(le - 1):
            prev_item = item_ids[pos]
            next_item = item_ids[pos + 1]

            for offset in [0, self.num_items, 2 * self.num_items]:
                u = node2idx[prev_item + offset]
                v = node2idx[next_item + offset]
                if u == v or adj[u][v] == 4:
                    continue
                if adj[v][u] == 2:  # 이미 역방향 존재 → bi-direction
                    adj[u][v] = 4
                    adj[v][u] = 4
                else:
                    adj[u][v] = 2  # out
                    adj[v][u] = 3  # in

        # ── alias 인덱스 (시퀀스 위치 → 노드 행렬 인덱스) ─────────────────
        alias_inputs = [node2idx[item] for item in item_ids]
        alias_inputs = alias_inputs + [0] * (self.max_seq_len - le)

        # ── node_pos_matrix: 각 노드가 시퀀스 어느 위치에 등장했는지 ───────
        u_input_padded = item_ids + [0] * (self.max_seq_len - le)
        node_pos_matrix = np.zeros((self.max_node_len, self.max_seq_len), dtype=np.float32)
        for n_idx, item in enumerate(u_nodes):
            pos_idx = [p for p, x in enumerate(u_input_padded) if x == item]
            node_pos_matrix[n_idx, pos_idx] = 1.0

        # item seq mask
        us_msks = [1] * le + [0] * (self.max_seq_len - le)

        return adj, nodes, node_type_mask, node_pos_matrix, alias_inputs, us_msks

    def __getitem__(self, idx):
        u, seq, target = self.samples[idx]

        # max_seq_len으로 truncate (최근 시퀀스 유지)
        seq      = seq[-self.max_seq_len:]
        item_ids = [s['item_id'] for s in seq]

        adj, nodes, node_type_mask, node_pos_matrix, alias_inputs, us_msks = \
            self._build_graph(item_ids)

        # negative 샘플링
        rated = self.train_item_dict[u] | {target}
        neg = np.random.randint(1, self.num_items)
        while neg in rated:
            neg = np.random.randint(1, self.num_items)

        return {
            'adj':             torch.tensor(adj,             dtype=torch.long),
            'nodes':           torch.tensor(nodes,           dtype=torch.long),
            'node_type_mask':  torch.tensor(node_type_mask,  dtype=torch.long),
            'node_pos_matrix': torch.tensor(node_pos_matrix, dtype=torch.float),
            'alias_inputs':    torch.tensor(alias_inputs,    dtype=torch.long),
            'us_msks':         torch.tensor(us_msks,         dtype=torch.long),
            'target':          torch.tensor(target,          dtype=torch.long),
            'negative':        torch.tensor(neg,             dtype=torch.long),
            'user_id':         torch.tensor(u,               dtype=torch.long),
        }
