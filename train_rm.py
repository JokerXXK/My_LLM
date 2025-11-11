# train_rm.py  ← 2025 终极稳定版（已修复所有维度、设备、数据对齐问题）
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, argparse, time, random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from models import HybridGNN
from data import DataBasicLoader
import logging
from tensorboardX import SummaryWriter

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='australia-covid')
    ap.add_argument('--sim_mat', type=str, default='australia-adj')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-5)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--window', type=int, default=20)
    ap.add_argument('--horizon', type=int, default=1)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--patience', type=int, default=50)

    # 奖励模型专用
    ap.add_argument('--is_reward_model', action='store_true', default=True)
    ap.add_argument('--predictor_path', type=str, default='predictor_save/best_hybridgnn_predictor.pt')
    ap.add_argument('--alpha_reward', type=float, default=100.0)
    ap.add_argument('--save_dir', type=str, default='rm_save')

    # HybridGNN 参数（必须与 train_predictor.py 一致！）
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--hidR', type=int, default=64)
    ap.add_argument('--hidA', type=int, default=64)
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--res', type=int, default=1)
    ap.add_argument('--n_hidden', type=int, default=64)
    ap.add_argument('--rnn_model', type=str, default='RNN')
    ap.add_argument('--residual_window', type=int, default=4)
    ap.add_argument('--ratio', type=float, default=1.0)

    return ap.parse_args()

# --- 生成 R_true 标签 ---
def generate_reward_labels(predictor_model, X_history, Y_true, alpha, device, batch_size=32):
    predictor_model.eval()
    mae_list = []

    with torch.no_grad():
        for i in range(0, len(X_history), batch_size):
            X_batch = X_history[i:i+batch_size].to(device)
            Y_batch = Y_true[i:i+batch_size].to(device)

            pred, _ = predictor_model(X_batch)
            mae = F.l1_loss(pred, Y_batch, reduction='none').mean(dim=1)  # [B]
            mae_list.extend(mae.cpu().numpy())

    mae_array = np.array(mae_list)
    R_true = 1.0 / (1.0 + alpha * mae_array)
    return torch.FloatTensor(R_true).unsqueeze(1).to(device)  # [N, 1]

# --- 训练/验证函数 ---
def train_rm(model, data_loader, data, R_true_labels, optimizer, device):
    model.train()
    total_loss = 0.0
    n_samples = 0

    # 关键修复 1: 必须同时取出 X_batch (历史) 和 Y_batch (真实目标)
    for X_batch, Y_batch in data_loader.get_batches(data, args.batch, shuffle=True):
        X_batch = X_batch.to(device) # [B, 20, M]
        
        # 关键修复 2: 将 Y_batch [B, M] 维度增加到 [B, 1, M]
        Y_batch = Y_batch.unsqueeze(1).to(device) # [B, 1, M]
        
        # 关键修复 3: 连接 X_batch 和 Y_batch 形成 [B, 21, M] 作为 RM 的输入
        X_rm_input = torch.cat([X_batch, Y_batch], dim=1) # [B, 21, M]

        batch_size = X_batch.size(0)
        R_true_batch = R_true_labels[n_samples:n_samples + batch_size].to(device)

        optimizer.zero_grad()
        R_pred, _ = model(X_rm_input) # 传入正确的 21 步输入
        loss = F.mse_loss(R_pred, R_true_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        n_samples += batch_size

    return total_loss / n_samples

def evaluate_rm(model, data_loader, data, R_true_labels, device):
    model.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        # 关键修复 1: 必须同时取出 X_batch (历史) 和 Y_batch (真实目标)
        for i, (X_batch, Y_batch) in enumerate(data_loader.get_batches(data, args.batch, shuffle=False)):
            X_batch = X_batch.to(device)
            
            # 关键修复 2: 将 Y_batch [B, M] 维度增加到 [B, 1, M]
            Y_batch = Y_batch.unsqueeze(1).to(device) 

            # 关键修复 3: 连接 X_batch 和 Y_batch 形成 [B, 21, M] 作为 RM 的输入
            X_rm_input = torch.cat([X_batch, Y_batch], dim=1) # [B, 21, M]
            
            batch_size = X_batch.size(0)
            start = i * args.batch
            end = start + batch_size
            R_true_batch = R_true_labels[start:end].to(device)

            R_pred, _ = model(X_rm_input) # 传入正确的 21 步输入
            loss = F.mse_loss(R_pred, R_true_batch)

            total_loss += loss.item() * batch_size
            n_samples += batch_size

    return total_loss / n_samples

# --- 主函数 ---
if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cuda = torch.cuda.is_available()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # TensorBoard
    log_token = f'RM.{args.dataset}.w-{args.window}.h-{args.horizon}.a-{args.alpha_reward}'
    writer = SummaryWriter(f'tensorboard/rm_{log_token}')

    print("加载数据 (RM 模式)...")
    args.is_reward_model = True
    data_loader = DataBasicLoader(args)
    train_data = data_loader.train
    val_data = data_loader.val

    print(f"训练样本: {len(train_data[0])}, 验证样本: {len(val_data[0])}")

    # 加载 Predictor
    print("加载 Predictor 模型...")
    pred_args = argparse.Namespace(**vars(args))
    pred_args.is_reward_model = False
    # 注意：这里需要重新初始化一个 data_loader，以确保 Predictor 使用正确的参数（特别是 residual_window）
    # 但由于 Predictor 只用于生成 R_true 标签，并且输入的是 X_history [B, 20, M]，
    # 我们可以直接使用 args，但为了安全起见，沿用您原始代码的逻辑。
    pred_loader = DataBasicLoader(pred_args) 

    # === 关键修复：智能加载 Predictor，自动忽略不匹配的层 ===
    predictor = HybridGNN(pred_args, pred_loader).to(device)

    state_dict = torch.load(args.predictor_path, map_location=device)
    model_dict = predictor.state_dict()

    # 只加载名字和形状都匹配的参数
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(state_dict)
    predictor.load_state_dict(model_dict)

    print(f"Predictor 智能加载成功！共加载 {len(state_dict)}/{len(model_dict)} 个参数")
    predictor.eval()

    # 生成 R_true 标签
    print("生成 R_true 标签...")
    X_train_hist = train_data[0][:, :args.window, :].to(device)
    Y_train_true = train_data[1].to(device)
    R_true_train = generate_reward_labels(predictor, X_train_hist, Y_train_true, args.alpha_reward, device)

    X_val_hist = val_data[0][:, :args.window, :].to(device)
    Y_val_true = val_data[1].to(device)
    R_true_val = generate_reward_labels(predictor, X_val_hist, Y_val_true, args.alpha_reward, device)

    print(f"R_true_train: min={R_true_train.min():.4f}, max={R_true_train.max():.4f}")
    print(f"R_true_val:   min={R_true_val.min():.4f}, max={R_true_val.max():.4f}")

    # 初始化 RM
    print("初始化 Reward Model...")
    model = HybridGNN(args, data_loader).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 训练循环
    print("开始训练 Reward Model...")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss = train_rm(model, data_loader, train_data, R_true_train, optimizer, device)
        val_loss = evaluate_rm(model, data_loader, val_data, R_true_val, device)

        writer.add_scalars('Loss/RM', {'train': train_loss, 'val': val_loss}, epoch)

        print(f"Epoch {epoch:3d} | {time.time()-start_time:5.1f}s | "
              f"train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(args.save_dir, 'best_hybridgnn_rm.pt')
            torch.save(model.state_dict(), save_path)
            print(f"    → 保存最佳模型: {save_path}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print("RM 训练完成！")
    writer.close()