# models.py  ← 2025 终极稳定版（已修复所有维度、残差、RM 输出问题）
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from layers import GraphConvLayer
from utils import getLaplaceMat, sparse_mx_to_torch_sparse_tensor, normalize_adj2
import numpy as np # 导入 numpy 用于矩阵处理


class HybridGNN(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.is_reward_model = getattr(args, 'is_reward_model', False)
        if self.is_reward_model:
            self.window_len = args.window + args.horizon # 20 + 1 = 21
        else:
            self.window_len = args.window # 20
        self.use_residual = getattr(args, 'use_residual', True)

        self.x_h = 1
        self.m = data.m
        self.w = args.window
        self.droprate = args.dropout
        self.dropout = nn.Dropout(self.droprate)

        self.hidR = args.hidR
        self.hidA = args.hidA
        self.n = args.n
        self.res = args.res
        self.n_hidden = args.n_hidden
        self.k = args.k

        # 邻接矩阵
        self.adj_orig = data.orig_adj
        
        # 核心修正：从 data.orig_adj 计算度向量并转换为 [1, M, 1] PyTorch 张量
        # 1. 计算度向量 (从 data.orig_adj)
        degree_vector_np = np.array(data.orig_adj.sum(axis=1)).flatten()
        # 2. 转换为 PyTorch 张量并调整形状为 [M, 1]
        degree_tensor = torch.from_numpy(degree_vector_np).float().unsqueeze(-1)
        # 3. 提升维度：增加批次维度，使其形状为 [1, M, 1]，便于在 forward 中 expand 批次
        self.register_buffer('degree', degree_tensor.unsqueeze(0))

        # 时间特征卷积
        self.conv_short = nn.Conv1d(1, self.k, self.w)
        long_kernel = self.w // 2
        self.conv_long = nn.Conv1d(1, self.k, long_kernel, dilation=2)
        self.h_SC_dim = self.k * 2  # short + long

        # Attention
        self.WQ = nn.Linear(self.h_SC_dim, self.hidA)
        self.WK = nn.Linear(self.h_SC_dim, self.hidA)
        self.t_enc = nn.Linear(1, self.hidR)
        self.s_enc = nn.Linear(1, self.hidR)

        # RNN for dynamic graph
        rnn_class = {'LSTM': nn.LSTM, 'GRU': nn.GRU, 'RNN': nn.RNN}[args.rnn_model]
        self.rnn = rnn_class(
            input_size=1, hidden_size=self.n_hidden, # input_size 修正为 1
            num_layers=getattr(args, 'n_layer', 1),
            dropout=self.droprate, batch_first=True,
            bidirectional=getattr(args, 'bi', False)
        )
        self.rnn_out_dim = self.n_hidden * (2 if getattr(args, 'bi', False) else 1)

        # Dynamic adj parameters
        self.V = Parameter(torch.Tensor(self.n_hidden))
        self.W1 = Parameter(torch.Tensor(self.n_hidden, self.n_hidden))
        self.W2 = Parameter(torch.Tensor(self.n_hidden, self.n_hidden))
        self.Wb = Parameter(torch.Tensor(self.m, self.m))
        self.bv = Parameter(torch.Tensor(1))
        self.b1 = Parameter(torch.Tensor(self.n_hidden))
        self.wb = Parameter(torch.Tensor(1))
        self.d_gate = Parameter(torch.ones(self.m, self.m))

        # 归一化静态邻接矩阵
        adj_norm = normalize_adj2(self.adj_orig.toarray())  # 直接转 numpy！
        self.adj_geo = sparse_mx_to_torch_sparse_tensor(adj_norm).to_dense()
        if args.cuda:
            self.adj_geo = self.adj_geo.cuda()

        # GNN layers
        initial_dim = self.h_SC_dim + self.hidR + self.hidR
        self.GNNBlocks = nn.ModuleList([
            GraphConvLayer(initial_dim, initial_dim) for _ in range(self.n)
        ])

        # 输出头
        gnn_out_dim = initial_dim * (self.n + 1) if self.res == 1 else initial_dim
        final_dim = gnn_out_dim + self.rnn_out_dim

        if self.is_reward_model:
            self.output = nn.Sequential(
                nn.Linear(final_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        else:
            self.output = nn.Linear(final_dim, 1)

        # 残差（仅预测器）
        self.residual_window = getattr(args, 'residual_window', 0) if not self.is_reward_model else 0
        if self.residual_window > 0:
            self.residual = nn.Linear(self.residual_window, 1)
        self.ratio = getattr(args, 'ratio', 1.0)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p, -0.1, 0.1)

    def forward(self, x, isEval=False):
        batch_size, seq_len, n_nodes = x.shape
        orig_x = x

        # === 1. 时间特征提取 (已修复: 统一 L_out, 避免维度不匹配) ===
        h_SC_list = []
        for i in range(n_nodes):
            node_seq = x[:, :, i:i+1].permute(0, 2, 1)  # [B, 1, L]
            
            # Conv1d outputs: [B, k, L_out]
            short = self.conv_short(node_seq) # [B, k, L_out_short]
            long = self.conv_long(node_seq)   # [B, k, L_out_long]
            
            # 修复：在时间维度 (dim=-1) 上进行全局平均池化 (GAP)
            short_pooled = torch.mean(short, dim=-1) # [B, k]
            long_pooled = torch.mean(long, dim=-1)   # [B, k]
            
            # 在通道维度 (dim=1) 上拼接，得到 [B, 2*k]
            combined = torch.cat([short_pooled, long_pooled], dim=1) 
            
            h_SC_list.append(combined) # [B, 2*k]
            
        h_SC = torch.stack(h_SC_list, dim=1)  # [B, M, h_SC_dim=2*k]
        h_SC = F.relu(h_SC)
        h_SC = self.dropout(h_SC)

        # === 2. Transmission Risk ===
        Q = self.WQ(h_SC)
        K = self.WK(h_SC)
        attn = torch.bmm(Q, K.transpose(1, 2))
        attn = F.normalize(attn, p=2, dim=-1)
        h_G = self.t_enc(attn.sum(dim=-1, keepdim=True))
        h_G = self.dropout(h_G)

        # d_degree 形状: [1, M, 1] -> 扩展到 [B, M, 1]
        d_degree = self.degree.expand(batch_size, -1, -1)
        h_L = self.s_enc(d_degree)
        h_L = self.dropout(h_L)

        # === 3. 序列特征提取 (RNN/GRU) (已修复: 重塑为 [B*M, L, 1]) ===
        # orig_x 形状: [B, L, M]
        
        # 1. 调整维度：从 [B, L, M] 变为 [B, M, L]
        rnn_in = orig_x.permute(0, 2, 1).contiguous() # [B, M, L]
        
        # 2. 增加特征维度 (1 维)
        rnn_in = rnn_in.unsqueeze(-1)    # [B, M, L, 1]
        
        # 3. 合并批次和节点维度，重塑为 [B*M, L, 1]
        rnn_in = rnn_in.reshape(-1, self.window_len, self.x_h) # [B*M, L, 1]
        
        rnn_out, _ = self.rnn(rnn_in)  # rnn_out: [B*M, L, hidR]

        # 4. 提取最后一个时间步的输出，并分离批次和节点维度
        last_hidden = rnn_out[:, -1, :] # [B*M, hidR]
        last_hidden = last_hidden.reshape(batch_size, n_nodes, -1) # [B, M, hidR]

        # Dynamic adj
        h_m = last_hidden.unsqueeze(2).expand(-1, -1, self.m, -1)
        h_w = last_hidden.unsqueeze(1).expand(-1, self.m, -1, -1)
        a = F.elu(torch.matmul(h_m, self.W1.t()) + torch.matmul(h_w, self.W2.t()) + self.b1)
        a = torch.matmul(a, self.V) + self.bv
        a = F.normalize(a, p=2, dim=1)

        c_gate = torch.sigmoid(torch.matmul(a, self.Wb) + self.wb)
        adj_geo_batched = self.adj_geo.unsqueeze(0).expand(batch_size, -1, -1)
        adj_dynamic = adj_geo_batched * c_gate + a * (1 - c_gate)

        d_mat = torch.bmm(d_degree, d_degree.transpose(1, 2))
        d_mat = torch.sigmoid(self.d_gate * d_mat)
        adj_final = adj_dynamic + torch.mul(d_mat, adj_geo_batched)
        laplace = getLaplaceMat(batch_size, self.m, adj_final)

        # === 4. GNN ===
        feats = torch.cat([h_SC, h_G, h_L], dim=-1)
        gnn_feats = [feats]
        for layer in self.GNNBlocks:
            feats = self.dropout(layer(feats, laplace))
            gnn_feats.append(feats)
        gnn_final = torch.cat(gnn_feats, dim=-1) if self.res == 1 else feats

        # === 5. 最终融合 + 输出 ===
        final_input = torch.cat([gnn_final, last_hidden], dim=-1)
        out = self.output(final_input).squeeze(-1)  # [B, M]

        if self.is_reward_model:
            reward = out.mean(dim=1, keepdim=True)  # [B, 1]
            return reward, None
        else:
            if self.residual_window > 0:
                res_input = orig_x[:, -self.residual_window:, :]
                res_input = res_input.permute(0, 2, 1).contiguous().view(batch_size * self.m, -1)
                res_out = self.residual(res_input).view(batch_size, self.m)
                out = out * self.ratio + res_out
            return out, None
        
        