import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from sklearn.cluster import KMeans


def load_our_data(interaction_path, item_path, title_npy_path, max_seq_len=50, K=10):
    """
    우리 데이터셋 로딩 및 전처리
    - leave-two-out 분할 (train: 마지막 2개 제외, valid: 뒤에서 2번째, test: 마지막)
    - image: video_feature K-means clustering
    - text:  title_emb K-means clustering
    return:
        train_data, test_data: VLGraph 형식 {'seqs', 'img_seqs', 'txt_seqs', 'targets'}
        num_items, image_cluster_feature, text_cluster_feature
        item_image_list, item_text_list
        user_train, user_valid, user_test, train_item_dict
    """
    interactions_df = pd.read_parquet(interaction_path).sort_values(['user_id', 'timestamp'])
    item_df         = pd.read_parquet(item_path).reset_index(drop=True)
    title_feat      = np.load(title_npy_path)  # (num_items, 384)

    # 1-based item_id
    interactions_df = interactions_df.copy()
    interactions_df['item_id'] = interactions_df['item_id'] + 1
    item_df = item_df.copy()
    item_df['item_id'] = item_df['item_id'] + 1
    num_items = int(item_df['item_id'].max())

    # ── image K-means ──────────────────────────────────────────────────────
    print(f"Image K-means (K={K})...")
    image_feat   = np.stack(item_df['video_feature'].values)
    img_kmeans   = KMeans(n_clusters=K, random_state=42, n_init=10)
    image_labels = img_kmeans.fit_predict(image_feat)
    image_cluster_feature = img_kmeans.cluster_centers_  # (K, feat_dim)

    # ── text K-means ───────────────────────────────────────────────────────
    print(f"Text K-means (K={K})...")
    txt_kmeans  = KMeans(n_clusters=K, random_state=42, n_init=10)
    text_labels = txt_kmeans.fit_predict(title_feat)
    text_cluster_feature = txt_kmeans.cluster_centers_   # (K, 384)

    # item → cluster 매핑 (1-based)
    item_ids = item_df['item_id'].values
    item_to_img_cluster = dict(zip(item_ids, image_labels + 1))  # 1-based
    item_to_txt_cluster = dict(zip(item_ids, text_labels  + 1))  # 1-based

    # item_image_list, item_text_list: (num_items+1, 1)
    item_image_list = np.zeros((num_items + 1, 1), dtype=np.int64)
    item_text_list  = np.zeros((num_items + 1, 1), dtype=np.int64)
    for iid in item_ids:
        item_image_list[iid, 0] = item_to_img_cluster.get(iid, 1)
        item_text_list[iid, 0]  = item_to_txt_cluster.get(iid, 1)

    # ── 유저 시퀀스 구성 ───────────────────────────────────────────────────
    user_sequences = defaultdict(list)
    for _, row in interactions_df.iterrows():
        u = int(row['user_id'])
        i = int(row['item_id'])
        user_sequences[u].append({
            'item_id':     i,
            'img_cluster': item_to_img_cluster.get(i, 1),
            'txt_cluster': item_to_txt_cluster.get(i, 1),
        })

    # ── leave-two-out 분할 ────────────────────────────────────────────────
    user_train = {}
    user_valid = {}
    user_test  = {}
    train_item_dict = defaultdict(set)

    for u, seq in user_sequences.items():
        if len(seq) < 3:
            user_train[u] = seq
            user_valid[u] = []
            user_test[u]  = []
        else:
            user_train[u] = seq[:-2]
            user_valid[u] = [seq[-2]]
            user_test[u]  = [seq[-1]]
        for s in user_train[u]:
            train_item_dict[u].add(s['item_id'])

    # ── VLGraph 형식 데이터 구성 ──────────────────────────────────────────
    # train: input=train_seq, target=valid_item
    # test:  input=train+valid_seq, target=test_item
    train_data = {'seqs': [], 'img_seqs': [], 'txt_seqs': [], 'targets': [], 'user_ids': []}
    test_data  = {'seqs': [], 'img_seqs': [], 'txt_seqs': [], 'targets': [], 'user_ids': []}

    for u, seq in user_sequences.items():
        if len(seq) < 3:
            continue

        train_seq  = seq[:-2]
        valid_item = seq[-2]
        test_item  = seq[-1]

        if len(train_seq) < 1:
            continue

        # train 샘플
        train_data['seqs'].append([s['item_id']     for s in train_seq][-max_seq_len:])
        train_data['img_seqs'].append([[s['img_cluster']] for s in train_seq][-max_seq_len:])
        train_data['txt_seqs'].append([[s['txt_cluster']] for s in train_seq][-max_seq_len:])
        train_data['targets'].append(valid_item['item_id'])
        train_data['user_ids'].append(u)

        # test 샘플: train + valid 시퀀스
        test_seq = train_seq + [valid_item]
        test_data['seqs'].append([s['item_id']     for s in test_seq][-max_seq_len:])
        test_data['img_seqs'].append([[s['img_cluster']] for s in test_seq][-max_seq_len:])
        test_data['txt_seqs'].append([[s['txt_cluster']] for s in test_seq][-max_seq_len:])
        test_data['targets'].append(test_item['item_id'])
        test_data['user_ids'].append(u)

    return (train_data, test_data, num_items,
            image_cluster_feature, text_cluster_feature,
            item_image_list, item_text_list,
            user_train, user_valid, user_test, train_item_dict)


class Data(Dataset):
    """
    VLGraph용 Dataset (우리 데이터 형식)
    - item 시퀀스 + image cluster 시퀀스 + text cluster 시퀀스
    - 그래프 구성: item/image/text 노드 + 10가지 엣지 타입
    """
    def __init__(self, data, num_items, max_len, link_k=1):
        self.data      = data
        self.num_items = num_items
        self.max_len   = max_len
        self.k         = link_k
        self.length    = len(data['seqs'])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        u_input     = self.data['seqs'][index]
        image_input = self.data['img_seqs'][index]
        text_input  = self.data['txt_seqs'][index]
        target      = self.data['targets'][index]

        le = len(u_input)

        # 노드 구성: item 노드(1) + image cluster 노드(2) + text cluster 노드(3)
        u_nodes = np.unique(u_input).tolist()
        i_nodes = np.unique([y for x in image_input for y in x]).tolist()
        t_nodes = np.unique([y for x in text_input  for y in x]).tolist()
        nodes   = u_nodes + i_nodes + t_nodes
        nodes   = np.asarray(nodes + (self.max_len - len(nodes)) * [0])

        node_type_mask = (
            [1] * len(u_nodes) +
            [2] * len(i_nodes) +
            [3] * len(t_nodes) +
            [0] * (self.max_len - len(u_nodes) - len(i_nodes) - len(t_nodes))
        )

        # 인접 행렬: 엣지 타입 (1~10)
        # self-loop(1), out(2), in(3), bi(4)
        # item→img(5), img→item(6), item→txt(7), txt→item(8), img→txt(9), txt→img(10)
        adj = np.zeros((self.max_len, self.max_len))

        for i in np.arange(le):
            item     = u_input[i]
            item_idx = np.where(nodes == item)[0][0]
            adj[item_idx][item_idx] = 1

            for img in image_input[i]:
                img_idx = np.where(nodes == img)[0][0]
                adj[img_idx][img_idx] = 1
                adj[item_idx][img_idx] = 5
                adj[img_idx][item_idx] = 6

            for txt in text_input[i]:
                txt_idx = np.where(nodes == txt)[0][0]
                adj[txt_idx][txt_idx] = 1
                adj[item_idx][txt_idx] = 7
                adj[txt_idx][item_idx] = 8

            for img in image_input[i]:
                for txt in text_input[i]:
                    img_idx = np.where(nodes == img)[0][0]
                    txt_idx = np.where(nodes == txt)[0][0]
                    adj[img_idx][txt_idx] = 9
                    adj[txt_idx][img_idx] = 10

        for i in np.arange(le - 1):
            prev_item = u_input[i]
            next_item = u_input[i + 1]
            u = np.where(nodes == prev_item)[0][0]
            v = np.where(nodes == next_item)[0][0]
            if u == v or adj[u][v] == 4:
                pass
            elif adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4
            else:
                adj[u][v] = 2
                adj[v][u] = 3

            for prev_img in image_input[i]:
                for next_img in image_input[i + 1]:
                    u = np.where(nodes == prev_img)[0][0]
                    v = np.where(nodes == next_img)[0][0]
                    if u == v or adj[u][v] == 4:
                        continue
                    if adj[v][u] == 2:
                        adj[u][v] = 4
                        adj[v][u] = 4
                    else:
                        adj[u][v] = 2
                        adj[v][u] = 3

            for prev_txt in text_input[i]:
                for next_txt in text_input[i + 1]:
                    u = np.where(nodes == prev_txt)[0][0]
                    v = np.where(nodes == next_txt)[0][0]
                    if u == v or adj[u][v] == 4:
                        continue
                    if adj[v][u] == 2:
                        adj[u][v] = 4
                        adj[v][u] = 4
                    else:
                        adj[u][v] = 2
                        adj[v][u] = 3

        alias_inputs = []
        for item in u_input:
            item_idx = np.where(nodes == item)[0][0]
            alias_inputs.append(item_idx)

        alias_img_inputs = [[0] * self.k for _ in range(self.max_len)]
        for i, img_bundle in enumerate(image_input):
            for j, img in enumerate(img_bundle[:self.k]):
                img_idx = np.where(nodes == img)[0][0]
                alias_img_inputs[i][j] = img_idx

        alias_txt_inputs = [[0] * self.k for _ in range(self.max_len)]
        for i, txt_bundle in enumerate(text_input):
            for j, txt in enumerate(txt_bundle[:self.k]):
                txt_idx = np.where(nodes == txt)[0][0]
                alias_txt_inputs[i][j] = txt_idx

        alias_inputs = alias_inputs + [0] * (self.max_len - le)
        u_input_pad  = list(u_input) + [0] * (self.max_len - le)
        us_msks      = [1] * le + [0] * (self.max_len - le) if le < self.max_len else [1] * self.max_len

        node_pos_matrix = np.zeros((self.max_len, self.max_len))
        n_idx = 0
        for item in u_nodes:
            pos_idx = [idx for idx, v in enumerate(u_input_pad) if v == item]
            node_pos_matrix[n_idx][pos_idx] = 1
            n_idx += 1
        for image in i_nodes:
            pos_idx = [idx for idx, sublist in enumerate(image_input) if image in sublist]
            node_pos_matrix[n_idx][pos_idx] = 1
            n_idx += 1
        for text in t_nodes:
            pos_idx = [idx for idx, sublist in enumerate(text_input) if text in sublist]
            node_pos_matrix[n_idx][pos_idx] = 1
            n_idx += 1

        return [torch.tensor(adj,              dtype=torch.float),
                torch.tensor(nodes,            dtype=torch.long),
                torch.tensor(node_type_mask,   dtype=torch.long),
                torch.tensor(node_pos_matrix,  dtype=torch.float),
                torch.tensor(us_msks,          dtype=torch.long),
                torch.tensor(target,           dtype=torch.long),
                torch.tensor(u_input_pad,      dtype=torch.long),
                torch.tensor(alias_inputs,     dtype=torch.long),
                torch.tensor(alias_img_inputs, dtype=torch.long),
                torch.tensor(alias_txt_inputs, dtype=torch.long)]
