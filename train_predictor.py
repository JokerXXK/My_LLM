# train_predictor.py  ← 2025 终极稳定版（已修复所有维度、路径、设备问题）
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from scipy.stats import pearsonr
from math import sqrt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from models import HybridGNN
from data import DataBasicLoader
from utils import peak_error

from tensorboardX import SummaryWriter

# ================== 参数 ==================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='australia-covid', help="Dataset string")
parser.add_argument('--sim_mat', type=str, default='australia-adj')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--window', type=int, default=20)
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--save_dir', type=str, default='predictor_save')

# HybridGNN 超参数（必须与 RM 一致！）
parser.add_argument('--k', type=int, default=8)
parser.add_argument('--hidR', type=int, default=64)
parser.add_argument('--hidA', type=int, default=64)
parser.add_argument('--n', type=int, default=2)
parser.add_argument('--res', type=int, default=1)
parser.add_argument('--n_hidden', type=int, default=64)
parser.add_argument('--rnn_model', type=str, default='RNN')
parser.add_argument('--residual_window', type=int, default=4)
parser.add_argument('--ratio', type=float, default=1.0)
parser.add_argument('--use_residual', type=bool, default=True)

args = parser.parse_args()

# ================== 环境设置 ==================
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.cuda = torch.cuda.is_available()
print(f"使用设备: {device}")

# ================== 数据加载 ==================
data_args = argparse.Namespace(
    dataset=args.dataset,
    sim_mat=args.sim_mat,
    window=args.window,
    horizon=args.horizon,
    train=0.5, val=0.2, test=0.3,
    cuda=args.cuda,
    batch=args.batch,
    is_reward_model=False,
    dropout=args.dropout,
    rnn_model=args.rnn_model,
    n_hidden=args.n_hidden,
    hidR=args.hidR,
    hidA=args.hidA,
    n=args.n,
    k=args.k,
    res=args.res,
    residual_window=args.residual_window,
    ratio=args.ratio,
    use_residual=args.use_residual
)
data_loader = DataBasicLoader(data_args)
print(f"训练集: {len(data_loader.train[0])}, 验证集: {len(data_loader.val[0])}, 测试集: {len(data_loader.test[0])}")

# ================== 模型 ==================
args.is_reward_model = False
model = HybridGNN(args, data_loader).to(device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# TensorBoard
os.makedirs(args.save_dir, exist_ok=True)
log_token = f'Pred.{args.dataset}.w-{args.window}.h-{args.horizon}'
writer = SummaryWriter(f'tensorboard/{log_token}')

# ================== 评估函数 ==================
def evaluate(data, tag='val'):
    model.eval()
    total_loss = 0.0
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for X_batch, Y_batch in data_loader.get_batches(data, args.batch, shuffle=False):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            pred, _ = model(X_batch)
            loss = F.l1_loss(pred, Y_batch)
            total_loss += loss.item() * X_batch.size(0)

            all_y_true.append(Y_batch.cpu())
            all_y_pred.append(pred.cpu())

    total_loss /= len(data[0])
    y_true = torch.cat(all_y_true).numpy()
    y_pred = torch.cat(all_y_pred).numpy()

    # 反归一化
    y_true_raw = y_true * (data_loader.max - data_loader.min) + data_loader.min
    y_pred_raw = y_pred * (data_loader.max - data_loader.min) + data_loader.min

    # 指标计算
    mae = mean_absolute_error(y_true_raw.flatten(), y_pred_raw.flatten())
    rmse = sqrt(mean_squared_error(y_true_raw.flatten(), y_pred_raw.flatten()))
    pcc = pearsonr(y_true_raw.flatten(), y_pred_raw.flatten())[0]
    r2 = r2_score(y_true_raw.flatten(), y_pred_raw.flatten())
    peak_mae = peak_error(y_true_raw.copy(), y_pred_raw.copy(), data_loader.peak_thold)

    return total_loss, mae, rmse, pcc, r2, peak_mae

# ================== 训练函数 ==================
def train(data):
    model.train()
    total_loss = 0.0
    for X_batch, Y_batch in data_loader.get_batches(data, args.batch, shuffle=True):
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        pred, _ = model(X_batch)
        loss = F.l1_loss(pred, Y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(data[0])

# ================== 主循环 ==================
print("开始训练 Predictor...")
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(1, args.epochs + 1):
    start_time = time.time()
    train_loss = train(data_loader.train)
    val_loss, val_mae, val_rmse, val_pcc, val_r2, val_peak = evaluate(data_loader.val)

    writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
    writer.add_scalars('Metrics/MAE', {'val': val_mae}, epoch)
    writer.add_scalars('Metrics/RMSE', {'val': val_rmse}, epoch)

    print(f"Epoch {epoch:4d} | {time.time()-start_time:5.1f}s | "
          f"train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | "
          f"MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | PCC: {val_pcc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_path = os.path.join(args.save_dir, 'best_hybridgnn_predictor.pt')
        torch.save(model.state_dict(), save_path)
        print(f"    → 保存最佳模型: {save_path}")

        # 测试集评估
        test_loss, test_mae, test_rmse, test_pcc, test_r2, test_peak = evaluate(data_loader.test, 'test')
        print(f"    TEST → MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | PCC: {test_pcc:.4f} | PeakMAE: {test_peak:.4f}")
    else:
        patience_counter += 1

    if patience_counter >= args.patience:
        print(f"Early stopping at epoch {epoch}")
        break

# ================== 最终测试 ==================
model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_hybridgnn_predictor.pt')))
test_loss, test_mae, test_rmse, test_pcc, test_r2, test_peak = evaluate(data_loader.test, 'test')
print("\n" + "="*60)
print("最终测试结果")
print(f"TEST MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | PCC: {test_pcc:.4f} | R2: {test_r2:.4f} | PeakMAE: {test_peak:.4f}")
print("="*60)

writer.close()
print(f"训练完成！最佳模型已保存: {args.save_dir}/best_hybridgnn_predictor.pt")