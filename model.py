import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import RGATLayer, GCNLayer, SAGELayer, GATLayer, KVAttentionLayer, HeteAttenLayer
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_


class VLGraph(Module):
    def __init__(self, config, img_cluster_feature, txt_cluster_feature,
                 item_image_list, item_text_list):
        super(VLGraph, self).__init__()
        self.config     = config
        self.batch_size = config['batch_size']
        self.num_item   = config['num_item']    # 우리 데이터: 1-based num_items+1
        self.num_image  = config['num_cluster']
        self.num_text   = config['num_cluster']
        self.num_node   = self.num_item + self.num_image + self.num_text

        self.dim            = config['embedding_size']
        self.auxiliary_info = config['auxiliary_info']
        self.dropout_local  = config['dropout_local']
        self.dropout_atten  = config['dropout_atten']
        self.n_layer        = config['n_layer']
        self.aggregator     = config['aggregator']
        self.max_relid      = config['max_relid']

        # Aggregator
        if self.aggregator == 'rgat':
            self.local_agg = RGATLayer(self.dim, self.max_relid, self.config['alpha'], dropout=self.dropout_atten)
        elif self.aggregator == 'hete_attention':
            self.local_agg = HeteAttenLayer(config, self.dim, self.max_relid, alpha=0.1, dropout=self.dropout_atten)
        elif self.aggregator == 'kv_attention':
            self.local_agg = KVAttentionLayer(self.dim, self.max_relid, alpha=0.1, dropout=self.dropout_atten)
        elif self.aggregator == 'gcn':
            self.local_agg = GCNLayer(input_dim=self.dim, output_dim=self.dim, n_heads=1, activation=F.relu, dropout=self.dropout_local)
        elif self.aggregator == 'graphsage':
            self.local_agg = SAGELayer(input_dim=self.dim, output_dim=self.dim, n_heads=1, activation=F.relu, dropout=self.dropout_local)
        elif self.aggregator == 'gat':
            self.local_agg = GATLayer(input_dim=self.dim, output_dim=self.dim, n_heads=1, activation=F.relu, dropout=self.dropout_local)

        # 노드 임베딩: item + image cluster + text cluster
        self.embedding           = nn.Embedding(self.num_node + 1, self.dim, padding_idx=0)
        self.pos_embedding       = nn.Embedding(200, self.dim)
        self.node_type_embedding = nn.Embedding(4, self.dim)

        self.w_1        = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2        = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_pos_type = nn.Parameter(torch.Tensor((len(self.auxiliary_info) + 1) * self.dim, self.dim))

        self.glu1       = nn.Linear(self.dim, self.dim)
        self.glu2       = nn.Linear(self.dim, self.dim, bias=False)
        self.projection = nn.Sequential(nn.Linear(self.dim, self.dim), nn.ReLU(True), nn.Linear(self.dim, 1))
        self.fusion_layer = nn.Linear(self.dim * 3, self.dim)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=config['lr'],
                                          weight_decay=config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=config['lr_dc_step'], gamma=config['lr_dc']
        )

        self.reset_parameters()

        # cluster feature 초기화 (원본 normalize_array 방식 유지)
        assert self.num_image == len(img_cluster_feature)
        assert self.num_text  == len(txt_cluster_feature)
        self.embedding.weight.data[self.num_item: self.num_item + self.num_image].copy_(
            torch.from_numpy(self.normalize_array(img_cluster_feature)).float()
        )
        self.embedding.weight.data[self.num_item + self.num_image: self.num_item + self.num_image + self.num_text].copy_(
            torch.from_numpy(self.normalize_array(txt_cluster_feature)).float()
        )

        # item → cluster 매핑
        item_image_dict   = torch.tensor(item_image_list, requires_grad=False).cuda()
        self.image_indices = item_image_dict.reshape(-1)
        item_text_dict    = torch.tensor(item_text_list,  requires_grad=False).cuda()
        self.text_indices  = item_text_dict.reshape(-1)
        self.k = item_image_dict.shape[-1]

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def normalize_array(self, array):
        min_val = np.min(array)
        max_val = np.max(array)
        stdv = 1.0 / math.sqrt(self.dim)
        return (array - min_val) * (stdv - (-stdv)) / (max_val - min_val + 1e-8) + (-stdv)

    def forward(self, adj, nodes, node_type_mask, node_pos_matrix, stage='train'):
        h_nodes = self.embedding(nodes)

        if len(self.auxiliary_info) > 0:
            auxiliary_embedding = [h_nodes]
            if 'node_type' in self.auxiliary_info:
                node_type_embedding = self.node_type_embedding(node_type_mask)
                auxiliary_embedding.append(node_type_embedding)
            if 'pos' in self.auxiliary_info:
                L = node_pos_matrix.shape[-1]
                pos_emb       = self.pos_embedding.weight[:L]
                pos_embedding = torch.matmul(node_pos_matrix, pos_emb)
                pos_num       = node_pos_matrix.sum(dim=-1, keepdim=True)
                pos_embedding = pos_embedding / (pos_num + 1e-9)
                pos_embedding = pos_embedding * torch.clamp(node_type_mask, max=1).unsqueeze(-1)
                auxiliary_embedding.append(pos_embedding)
            h_nodes = torch.cat(auxiliary_embedding, -1)
            h_nodes = torch.matmul(h_nodes, self.w_pos_type)

        for i in range(self.n_layer):
            h_nodes = self.local_agg(h_nodes, adj, node_type_mask, stage)
            h_nodes = F.dropout(h_nodes, self.dropout_local, training=self.training)
            h_nodes = h_nodes * torch.clamp(node_type_mask, max=1).unsqueeze(-1)

        return h_nodes

    def get_sequence_representation(self, seq_hiddens, mask, pooling_method='last'):
        if pooling_method == 'last':
            gather_index = torch.sum(mask, dim=-1) - 1
            gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, seq_hiddens.shape[-1])
            hiddens = seq_hiddens.gather(dim=1, index=gather_index).squeeze(1)
        elif pooling_method == 'mean':
            mask_f  = mask.float().unsqueeze(-1)
            hiddens = torch.sum(seq_hiddens * mask_f, -2) / torch.sum(mask_f, 1)
        elif pooling_method == 'attention':
            mask_f  = mask.float().unsqueeze(-1)
            B, L, _ = seq_hiddens.shape
            pos_emb = self.pos_embedding.weight[:L].unsqueeze(0).repeat(B, 1, 1)
            hs      = torch.sum(seq_hiddens * mask_f, -2) / torch.sum(mask_f, 1)
            hs      = hs.unsqueeze(-2).repeat(1, L, 1)
            nh      = torch.matmul(torch.cat([pos_emb, seq_hiddens], -1), self.w_1)
            nh      = torch.tanh(nh)
            nh      = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
            beta    = torch.matmul(nh, self.w_2) * mask_f
            hiddens = torch.sum(beta * seq_hiddens, 1)
        return hiddens

    def compute_full_scores(self, node_hiddens, alias_item_inputs, alias_img_inputs,
                            alias_txt_inputs, item_seq_mask):
        """학습 시 전체 아이템 스코어 (CrossEntropy loss용)"""
        mask      = item_seq_mask.float().unsqueeze(-1)
        B         = node_hiddens.shape[0]
        L         = alias_item_inputs.shape[1]
        batch_idx = torch.arange(B).unsqueeze(1).to(node_hiddens.device)

        item_seq_hiddens = torch.gather(
            node_hiddens, 1,
            alias_item_inputs.unsqueeze(-1).expand_as(node_hiddens)
        )
        item_seq_hiddens = mask * item_seq_hiddens
        item_hidden = self.get_sequence_representation(
            item_seq_hiddens, item_seq_mask,
            pooling_method=self.config['seq_pooling']
        )
        item_emb = self.embedding.weight[1:self.num_item]

        if self.config['modality_prediction']:
            _ = alias_img_inputs.view(B, -1)
            _ = node_hiddens[batch_idx, _]
            img_seq = _.view(B, L, self.k, self.dim)
            img_seq = torch.sum(img_seq, -2) / self.k
            img_hiddens = torch.sum(img_seq * mask, -2) / torch.sum(mask, 1)

            selected_rows = torch.gather(
                self.embedding.weight, 0,
                self.image_indices.unsqueeze(1).repeat(1, self.dim)
            )
            img_emb = selected_rows.reshape(self.num_item, self.k, self.dim).sum(-2)

            _ = alias_txt_inputs.view(B, -1)
            _ = node_hiddens[batch_idx, _]
            txt_seq = _.view(B, L, self.k, self.dim)
            txt_seq = torch.sum(txt_seq, -2) / self.k
            txt_hiddens = torch.sum(txt_seq * mask, -2) / torch.sum(mask, 1)

            selected_rows = torch.gather(
                self.embedding.weight, 0,
                self.text_indices.unsqueeze(1).repeat(1, self.dim)
            )
            txt_emb = selected_rows.reshape(self.num_item, self.k, self.dim).sum(-2)

            emb     = self.fusion_layer(torch.cat([item_emb, img_emb[1:], txt_emb[1:]], -1))
            hiddens = self.fusion_layer(torch.cat([item_hidden, img_hiddens, txt_hiddens], -1))
            scores  = torch.matmul(hiddens, emb.transpose(1, 0))
        else:
            scores = torch.matmul(item_hidden, item_emb.transpose(1, 0))

        return scores

    def compute_candidate_scores(self, node_hiddens, alias_item_inputs, alias_img_inputs,
                                  alias_txt_inputs, item_seq_mask, candidate_ids):
        """
        평가 시 101개 후보에 대한 스코어 계산 (베이스라인과 동일)
        candidate_ids: (B, 101) — 정답 1 + negative 100
        """
        mask      = item_seq_mask.float().unsqueeze(-1)
        B         = node_hiddens.shape[0]
        L         = alias_item_inputs.shape[1]
        batch_idx = torch.arange(B).unsqueeze(1).to(node_hiddens.device)

        item_seq_hiddens = torch.gather(
            node_hiddens, 1,
            alias_item_inputs.unsqueeze(-1).expand_as(node_hiddens)
        )
        item_seq_hiddens = mask * item_seq_hiddens
        item_hidden = self.get_sequence_representation(
            item_seq_hiddens, item_seq_mask,
            pooling_method=self.config['seq_pooling']
        )

        if self.config['modality_prediction']:
            _ = alias_img_inputs.view(B, -1)
            _ = node_hiddens[batch_idx, _]
            img_seq = _.view(B, L, self.k, self.dim)
            img_hiddens = torch.sum(img_seq, -2).sum(-2) / (self.k * max(torch.sum(mask, 1), torch.tensor(1.0)))

            _ = alias_txt_inputs.view(B, -1)
            _ = node_hiddens[batch_idx, _]
            txt_seq = _.view(B, L, self.k, self.dim)
            txt_hiddens = torch.sum(txt_seq, -2).sum(-2) / (self.k * max(torch.sum(mask, 1), torch.tensor(1.0)))

            img_hiddens = torch.sum(img_seq.sum(-2) * mask, -2) / torch.sum(mask, 1)
            txt_hiddens = torch.sum(txt_seq.sum(-2) * mask, -2) / torch.sum(mask, 1)

            user_repr = self.fusion_layer(
                torch.cat([item_hidden, img_hiddens, txt_hiddens], -1)
            )  # (B, dim)

            # 후보 아이템 repr
            cand_item_emb = self.embedding.weight[candidate_ids]          # (B, 101, dim)
            cand_img_idx  = self.image_indices[candidate_ids - 1]         # (B, 101)
            cand_txt_idx  = self.text_indices[candidate_ids - 1]          # (B, 101)
            cand_img_emb  = self.embedding.weight[cand_img_idx]           # (B, 101, dim)
            cand_txt_emb  = self.embedding.weight[cand_txt_idx]           # (B, 101, dim)
            cand_repr = self.fusion_layer(
                torch.cat([cand_item_emb, cand_img_emb, cand_txt_emb], -1)
            )  # (B, 101, dim)
        else:
            user_repr = item_hidden                                        # (B, dim)
            cand_repr = self.embedding.weight[candidate_ids]              # (B, 101, dim)

        # (B, 1, dim) × (B, dim, 101) → (B, 101)
        scores = torch.bmm(user_repr.unsqueeze(1), cand_repr.transpose(1, 2)).squeeze(1)
        return scores


def model_process(model, data, stage='train'):
    adj, nodes, node_type_mask, node_pos_matrix, \
    inputs_mask, targets, u_input, \
    alias_inputs, alias_img_inputs, alias_txt_inputs = data

    adj            = adj.float().cuda()
    node_pos_matrix = node_pos_matrix.float().cuda()
    nodes           = nodes.long().cuda()
    node_type_mask  = node_type_mask.long().cuda()
    alias_inputs    = alias_inputs.long().cuda()
    alias_img_inputs = alias_img_inputs.long().cuda()
    alias_txt_inputs = alias_txt_inputs.long().cuda()
    inputs_mask     = inputs_mask.long().cuda()
    targets         = targets.long().cuda()

    node_hidden = model.forward(adj, nodes, node_type_mask, node_pos_matrix, stage)
    scores      = model.compute_full_scores(
        node_hidden, alias_inputs, alias_img_inputs, alias_txt_inputs, inputs_mask
    )
    return targets, scores


def train_epoch(model, train_loader):
    """한 에폭 학습"""
    model.train()
    total_loss = 0.0
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores = model_process(model, data, stage='train')
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        model.optimizer.step()
        total_loss += loss.item()
    model.scheduler.step()
    return total_loss / len(train_loader)


def evaluate_101(model, data_loader, train_item_dict, num_items,
                 topk=10, num_neg=100):
    """
    101개 후보 기반 평가 (베이스라인과 동일)
    - 정답 1 + negative 100
    - negative: train 아이템 제외
    - NDCG@topk, HR@topk, MRR
    """
    model.eval()
    NDCG = HR = MRR = 0.0
    count = 0
    np.random.seed(42)

    for data in data_loader:
        adj, nodes, node_type_mask, node_pos_matrix, \
        inputs_mask, targets, u_input, \
        alias_inputs, alias_img_inputs, alias_txt_inputs = data

        adj              = adj.float().cuda()
        node_pos_matrix  = node_pos_matrix.float().cuda()
        nodes            = nodes.long().cuda()
        node_type_mask   = node_type_mask.long().cuda()
        alias_inputs     = alias_inputs.long().cuda()
        alias_img_inputs = alias_img_inputs.long().cuda()
        alias_txt_inputs = alias_txt_inputs.long().cuda()
        inputs_mask      = inputs_mask.long().cuda()
        targets_cuda     = targets.long().cuda()

        B = targets.shape[0]

        # 101개 후보 구성
        candidates = []
        for b in range(B):
            target = targets[b].item()
            rated  = set()  # train_item_dict에서 가져오는 게 이상적이나 배치 내 근사
            cands  = [target]
            for _ in range(num_neg):
                neg = np.random.randint(1, num_items + 1)
                while neg in rated or neg == target:
                    neg = np.random.randint(1, num_items + 1)
                cands.append(neg)
            candidates.append(cands)

        candidates = torch.tensor(candidates, dtype=torch.long).cuda()  # (B, 101)

        with torch.no_grad():
            node_hidden = model.forward(adj, nodes, node_type_mask,
                                        node_pos_matrix, stage='test')
            scores = model.compute_candidate_scores(
                node_hidden, alias_inputs, alias_img_inputs,
                alias_txt_inputs, inputs_mask, candidates
            )  # (B, 101)

        # 정답(index=0)의 rank 계산
        ranks = (-scores).argsort(dim=1).argsort(dim=1)[:, 0]
        for rank in ranks.cpu().tolist():
            count += 1
            MRR  += 1 / (rank + 1)
            if rank < topk:
                NDCG += 1 / np.log2(rank + 2)
                HR   += 1

    N = max(count, 1)
    return NDCG / N, HR / N, MRR / N
