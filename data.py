# data.py  ← 宇宙级万能版：任意 txt、任意列数、永不改第二次！
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import scipy.sparse as sp
import os

class DataBasicLoader(object):
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.sim_mat = args.sim_mat
        self.window = args.window
        self.horizon = args.horizon
        self.cuda = args.cuda
        self.batch_size = args.batch

        # === 万能读取：自动识别 .txt 格式和列数 ===
        txt_path = f'data/{self.dataset}.txt'
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"\n找不到数据文件：{txt_path}\n请放入 data/ 文件夹！")

        print(f"正在读取数据：{txt_path}")
        
        # 智能读取：先读一行判断分隔符
        with open(txt_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        if ',' in first_line:
            raw_data = np.loadtxt(txt_path, delimiter=',')
        elif '\t' in first_line:
            raw_data = np.loadtxt(txt_path, delimiter='\t')
        else:
            raw_data = np.loadtxt(txt_path)  # 空格分隔

        raw_data = raw_data.astype(np.float32)
        print(f"原始数据形状: {raw_data.shape} → [时间步={raw_data.shape[0]}, 地区数={raw_data.shape[1]}]")

        self.raw_data = raw_data
        self.T, self.m = raw_data.shape  # m 自动等于列数！
        print(f"自动识别地区数量：{self.m} 个地点")

        # === 自动生成邻接矩阵（如果没有）===
        adj_path = f'data/{self.sim_mat}.txt'
        if not os.path.exists(adj_path):
            print(f"未找到邻接矩阵，自动生成全连接图 ({self.m}×{self.m})")
            adj = np.ones((self.m, self.m)) - np.eye(self.m)
            os.makedirs('data', exist_ok=True)
            np.savetxt(adj_path, adj, fmt='%.1f', delimiter=',')  # 加了这句！
            print(f"已生成: {adj_path}")

        # 读取邻接矩阵（自动识别逗号或空格）
        with open(adj_path, 'r') as f:
            sample = f.readline()
        delimiter = ',' if ',' in sample else None
        adj = np.loadtxt(adj_path, delimiter=delimiter)  # 智能读取！

        if adj.shape != (self.m, self.m):
            print(f"邻接矩阵尺寸不匹配！自动调整为 {self.m}×{self.m}")
            adj = np.ones((self.m, self.m)) - np.eye(self.m)
            np.savetxt(adj_path, adj, fmt='%.1f')

        self.orig_adj = sp.csr_matrix(adj)
        self.degree_adj = sp.csr_matrix(
            (np.ones(adj.shape[0]), (np.arange(self.m), np.arange(self.m))),
            shape=(self.m, self.m)
        )

        # === 标准化 ===
        self.min = raw_data.min()
        self.max = raw_data.max()
        self.data = (raw_data - self.min) / (self.max - self.min + 1e-8)

        # === 划分训练/验证/测试 ===
        self.train_ratio = getattr(args, 'train', 0.5)
        self.val_ratio = getattr(args, 'val', 0.2)

        train_end = int(self.T * self.train_ratio)
        val_end = int(self.T * (self.train_ratio + self.val_ratio))

        self.train_data = self._build_samples(0, train_end)
        self.val_data = self._build_samples(train_end, val_end)
        self.test_data = self._build_samples(val_end, self.T)

        # 关键修复：添加别名，兼容所有 train_*.py
        self.train = self.train_data
        self.val = self.val_data
        self.test = self.test_data

        self.peak_thold = np.percentile(self.raw_data, 95)

        print(f"样本数量 → 训练: {len(self.train_data[0])}, 验证: {len(self.val_data[0])}, 测试: {len(self.test_data[0])}")
        print("数据加载完成！准备起飞！")

    def _build_samples(self, start, end):
        X_list, Y_list = [], []
        for i in range(start, end - self.window - self.horizon + 1):
            x = self.data[i:i + self.window]      # [window, m]
            y = self.data[i + self.window:i + self.window + self.horizon]  # [horizon, m]
            if x.shape[0] == self.window and y.shape[0] == self.horizon:
                X_list.append(torch.from_numpy(x))
                Y_list.append(torch.from_numpy(y))
        
        if len(X_list) == 0:
            return [torch.zeros(0, self.window, self.m), torch.zeros(0, self.horizon, self.m)]
        
        X = torch.stack(X_list)  # [N, window, m]
        Y = torch.stack(Y_list)  # [N, horizon, m]
        return [X, Y.mean(dim=1)]  # 如果 horizon=1，压缩成 [N, m]

    def get_batches(self, data, batch_size, shuffle=True):
        X = data[0]
        Y = data[1]
        
        # 过滤空样本
        mask = X.sum(dim=(1,2)) > 1e-6
        X = X[mask]
        Y = Y[mask]
        
        if len(X) == 0:
            return

        idx = torch.randperm(len(X)) if shuffle else torch.arange(len(X))
        
        for i in range(0, len(X), batch_size):
            batch_idx = idx[i:i + batch_size]
            batch_X = X[batch_idx]
            batch_Y = Y[batch_idx]
            
            if self.cuda:
                batch_X = batch_X.cuda()
                batch_Y = batch_Y.cuda()
                
            yield [batch_X, batch_Y]