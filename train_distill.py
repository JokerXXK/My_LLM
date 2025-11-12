# train_distill.py  ← 2025 终极稳定版（已修复稳定性问题 & 添加诊断）
# 说明：
#  - 1) 使用 args.window 替换硬编码
#  - 2) create_prompt 动态生成 Day 1…Day {window} -> Day {window+1}
#  - 3) test_X 切片使用 args.window
#  - 4) RM 输入拼接历史 (window) + LLM 预测 (1) -> window+1 步
#  - 5) 增强稳定性：更稳健的 decode（只 decode 新生成 token），use_cache=False，
#       扩大 top_k，P_llm 尺度映射，SFT lr 降低，关闭 fp16 并添加 max_grad_norm。
#  - 6) 增加丰富的诊断打印（rewards 分布、answers、部分 X_rm_input）
# =================================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ATTN_IMPLEMENTATION"] = "eager"
import torch
import numpy as np
import re
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from data import DataBasicLoader
from models import HybridGNN
# 关键补丁（保留你原来的）
from transformers.cache_utils import DynamicCache
DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

# -------------------- 参数 --------------------
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default='australia-covid')
parser.add_argument('--sim_mat', type=str, default='australia-adj')
parser.add_argument('--window', type=int, default=20)          # ← 可变的历史长度
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--residual_window', type=int, default=4)
parser.add_argument('--use_residual', type=bool, default=True)
parser.add_argument('--predictor_path', type=str, default='predictor_save/best_hybridgnn_predictor.pt')
parser.add_argument('--rm_path', type=str, default='rm_save/best_hybridgnn_rm.pt')
parser.add_argument('--top_k', type=int, default=8, help="每代用于 SFT 的 top_k 候选数（默认8）。")
parser.add_argument('--gen_batch', type=int, default=4, help="生成时的 batch size")
parser.add_argument('--num_gen_rounds', type=int, default=2, help="对同一 batch 生成几轮不同候选以增加多样性")
parser.add_argument('--max_new_tokens', type=int, default=256)
parser.add_argument('--debug', action='store_true', help="开启更多调试打印")
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ================== 1. 加载数据 ==================
data_args = argparse.Namespace(
    dataset=args.dataset,
    sim_mat=args.sim_mat,
    window=args.window,          # ← 同步使用可变 window
    horizon=args.horizon,
    train=0.5, val=0.2, test=0.3,
    cuda=True, batch=128,
    is_reward_model=False,
    dropout=0.2, rnn_model='RNN', n_hidden=64,
    hidR=64, hidA=64, n=2, k=8, res=1
)
data_loader = DataBasicLoader(data_args)
num_regions = data_loader.m
print(f"数据加载完成：地区数 num_regions={num_regions}")

# ================== 2. 加载 TinyLlama 4bit ==================
LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"加载 {LLM_MODEL_ID} (4bit)...")
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    device_map={'': device},
    attn_implementation="eager",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    ),
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
print("LLM 与 LoRA 加载完成。")

# ================== 3. 加载 Predictor (修复尺寸不匹配) ==================
pred_args = argparse.Namespace(**{
    **vars(data_args),
    'is_reward_model': False,
    'window': args.window,
    'horizon': args.horizon,
    'cuda': True,
    'dropout': 0.2,
    'rnn_model': 'RNN',
    'n_hidden': 64,
    'hidR': 64,
    'hidA': 64,
    'n': 2,
    'k': 8,
    'res': 1,
    'residual_window': args.residual_window,
    'ratio': 1.0,
    'use_residual': args.use_residual
})
predictor = HybridGNN(pred_args, data_loader).to(device)
state_dict = torch.load(args.predictor_path, map_location=device)

# --- 智能加载 Predictor，过滤掉尺寸不匹配的参数 ---
model_dict = predictor.state_dict()
filtered_state_dict = {}
mismatched_keys = []
for k, v in state_dict.items():
    if k in model_dict and v.shape == model_dict[k].shape:
        filtered_state_dict[k] = v
    elif k in model_dict:
        mismatched_keys.append(f"{k} (Checkpoint:{v.shape} vs Model:{model_dict[k].shape})")
        continue
load_result = predictor.load_state_dict(filtered_state_dict, strict=False)
print("Predictor 智能加载成功 (非兼容权重已重新初始化)。")
if load_result.missing_keys:
    print(f"  - 缺失键 (已重新初始化): {load_result.missing_keys}")
if mismatched_keys:
    print(f"  - 重新初始化键 (维度不匹配，已使用当前数据集 {num_regions} 个地区的尺寸): {mismatched_keys}")
predictor.eval()

# ================== 4. 加载 Reward Model ==================
rm_args = argparse.Namespace(**{
    **vars(data_args),
    'is_reward_model': True,
    'window': args.window,          # ← 同步使用可变 window
    'cuda': True,
    'residual_window': 0
})
reward_model = HybridGNN(rm_args, data_loader).to(device)
reward_model.load_state_dict(torch.load(args.rm_path, map_location=device))
reward_model.eval()
print("Reward Model 加载成功")

# ================== 5. 生成提示（动态 window） ==================
def create_prompt(history_window):
    """
    history_window: shape (window, num_regions) or (window, num_regions) numpy array/list
    """
    lines = [f"Day {t+1}: {' '.join([f'{x:.6f}' for x in day])}"
             for t, day in enumerate(history_window)]
    history_text = '\n'.join(lines)
    return f"""<|user|>
{history_text}
Day {args.window + 1} (predict {num_regions} regions, space separated):<|end|>
<|assistant|>"""

# 取前 16 条测试样本的历史（长度 = args.window）
test_X = data_loader.test[0][:16]                     # (16, total_steps, num_regions)
histories = test_X[:, :args.window].cpu().numpy()    # (16, window, num_regions)
prompts = [create_prompt(hist) for hist in histories]
print(f"生成 {len(prompts)} 个 prompts，用于推理与自进化循环。")

# ================== 6. 自进化主循环 ==================
BATCH_SIZE_GEN = args.gen_batch
MAX_NEW_TOKENS = args.max_new_tokens
NUM_GEN_ROUNDS = args.num_gen_rounds
TOP_K = args.top_k

def parse_nums(text, num_regions):
    # 更稳健的数字提取：匹配浮点/整数，返回 num_regions 长度
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    nums = [float(x) for x in nums]
    if len(nums) < num_regions:
        nums.extend([0.0] * (num_regions - len(nums)))
    return nums[:num_regions]

# small util: safe decode tokens only (skip prompt)
def decode_generated_only(tokenizer, out_ids, in_len):
    new_ids = out_ids[in_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text

for gen in range(1, 6):
    print(f"\n{'='*20} 第 {gen}/5 代自进化 {'='*20}")

    # 在自进化循环的开头（每代开始前）加上：
    model = model.to(device)   # ← 关键修复！
    model.eval()

    # ---------- 1. 分批生成（并为每个 prompt 生成多轮以增加多样性） ----------
    responses = []   # will hold one answer per prompt per round
    responses_meta = []  # store (prompt_idx, round_idx)
    for i in range(0, len(prompts), BATCH_SIZE_GEN):
        batch_prompts = prompts[i:i + BATCH_SIZE_GEN]
        # tokenize inputs
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)
        # get per-sample input lens (count non-pad tokens as approximate length)
        input_lens = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1).tolist()

        # generate multiple rounds for diversity
        for r in range(NUM_GEN_ROUNDS):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.9 if r>0 else 0.7,  # first round slightly lower temp
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False,  # 明确禁用 cache 以避免和 gradient_checkpointing 警告
                )
            # outputs shape: (B, seq_len_out). We decode only the newly generated tokens.
            # If num_return_sequences is not used, outputs length equals batch size.
            for out_ids, in_len, prompt_idx in zip(outputs, input_lens, range(i, i + len(batch_prompts))):
                text_only = decode_generated_only(tokenizer, out_ids, in_len)
                responses.append(text_only)
                responses_meta.append((prompt_idx, r))
            torch.cuda.empty_cache()

    # responses now has many candidates (len = num_prompts * NUM_GEN_ROUNDS)
    # We need to map them back to prompts when scoring. For simplicity, we'll aggregate per prompt:
    # Build per-prompt candidate lists
    per_prompt_candidates = [[] for _ in range(len(prompts))]
    for (prompt_idx, round_idx), resp in zip(responses_meta, responses):
        per_prompt_candidates[prompt_idx].append(resp)

    # ---------- 2. 解析每个候选并用 RM 打分 ----------
    all_candidates = []   # (prompt_idx, candidate_text)
    for p_idx, cands in enumerate(per_prompt_candidates):
        for cand in cands:
            all_candidates.append((p_idx, cand))

    # Build P_llm for all candidates in batch (we must preserve order)
    candidate_texts = [cand for (_, cand) in all_candidates]
    candidate_prompt_indices = [p for (p, _) in all_candidates]

    # Parse numbers for every candidate
    parsed = [parse_nums(txt, num_regions) for txt in candidate_texts]  # list of lists
    P_llm_raw = torch.tensor(parsed, dtype=torch.float32).to(device)  # (N_candidates, num_regions)
    # reshape to (N,1,num_regions) to match concatenation dims later
    P_llm_raw = P_llm_raw.unsqueeze(1)

    # ---------------- 3. 将 P_llm 做尺度映射（把 LLM 输出映射回历史尺度） ----------------
    # 使用每条样本最后一个历史日作为尺度参考（per-prompt）
    # For each candidate, find its prompt idx → use test_X[prompt_idx, args.window-1] as scale
    ref_last_hist = test_X[candidate_prompt_indices, args.window-1].to(device)  # (N, num_regions)
    eps = 1e-6
    scale = ref_last_hist.abs() + eps  # (N, num_regions)
    # Normalize raw predictions to 0-1 per-sample, then scale
    p_raw = P_llm_raw.squeeze(1)  # (N, num_regions)
    p_min = p_raw.min(dim=1, keepdim=True).values
    p_max = p_raw.max(dim=1, keepdim=True).values
    denom = (p_max - p_min).clamp(min=1e-9)
    p_norm = (p_raw - p_min) / denom  # (N, num_regions)
    P_llm_scaled = (p_norm * scale).unsqueeze(1)  # (N,1,num_regions)

    # ---------------- 4. 给每个 candidate 计算 Reward（分批送入 RM） ----------------
    # We will process candidates in batches to avoid OOM
    rm_batch = 32
    rewards_list = []
    # For each candidate we need to construct X_rm_input = history(window) + candidate_prediction(1)
    # For a candidate belonging to prompt p_idx, the history is test_X[p_idx, :args.window]
    for start in range(0, P_llm_scaled.shape[0], rm_batch):
        end = start + rm_batch
        batch_P = P_llm_scaled[start:end].to(device)  # (B,1,num_regions)
        batch_prompt_idxs = candidate_prompt_indices[start:end]
        # stack corresponding histories
        batch_histories = test_X[batch_prompt_idxs, :args.window].to(device)  # (B, window, num_regions)
        X_rm_input = torch.cat([batch_histories, batch_P], dim=1)  # (B, window+1, num_regions)
        # diagnostics: print first batch's small slice
        if args.debug and start == 0:
            print("DEBUG: X_rm_input[0,:,:5] (first candidate, first 5 regions):")
            print(X_rm_input[0, :, :5].cpu().numpy())
        with torch.no_grad():
            batch_rewards, _ = reward_model(X_rm_input)
        batch_rewards = batch_rewards.squeeze(1).cpu().numpy()  # (B,)
        rewards_list.extend(batch_rewards.tolist())

    rewards = np.array(rewards_list)  # (N_candidates,)
    # Map rewards back to per-prompt lists
    per_prompt_rewards = [[] for _ in range(len(prompts))]
    for (p_idx, _), r in zip(all_candidates, rewards):
        per_prompt_rewards[p_idx].append(r)

    # ---------- 5. 为每个 prompt 选择 top_k 候选（基于 z-score 归一化的 reward） ----------
    # 我们将先对所有 candidate reward 做全局 z-score，再选 top_k across all candidates (全局最优)
    r_mean = rewards.mean()
    r_std = rewards.std() + 1e-9
    rewards_z = (rewards - r_mean) / r_std
    # 选全局 top_k（这样 SFT 使用的是全体候选的 top_k 而不是每 prompt 固定数量）
    K = min(TOP_K, len(rewards_z))
    top_global_idx = np.argsort(rewards_z)[-K:]  # indices into all_candidates
    # 获取 top_k 的对应 prompt index 和文本
    top_idx_info = [all_candidates[i] for i in top_global_idx]
    top_texts = [candidate_texts[i] for i in top_global_idx]
    top_rewards = rewards[top_global_idx]

    print("本代选出的 Top 候选（全局 Top-K）:")
    for ti, (pidx, _) in enumerate(top_idx_info):
        print(f"  Rank {ti+1}: prompt#{pidx}, reward={top_rewards[ti]:.6f}")
        if args.debug:
            print("    text:", top_texts[ti][:200].replace('\n', ' '))

    # ---------- 6. 诊断打印（整体） ----------
    print(">> rewards 全局分布: mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}".format(
        rewards.mean(), rewards.std(), rewards.min(), rewards.max()
    ))
    if args.debug:
        # 展示部分 raw answers 与对应 reward
        print(">> 示例候选与 reward（前8）:")
        for i in range(min(8, len(candidate_texts))):
            print(f"  cand[{i}] prompt={candidate_prompt_indices[i]} reward={rewards[i]:.6f} text={candidate_texts[i][:120].replace(chr(10),' ')}")

    # ---------- 7. 构造 SFT 数据（扩大 top_k, 并做简单重复以扩大训练量） ----------
    train_texts = []
    for idx in top_global_idx:
        p_idx, _ = all_candidates[idx]
        cand_text = candidate_texts[idx]
        # 构造训练示例：prompt + candidate + end token
        full_text = prompts[p_idx] + cand_text + "<|end|>"
        train_texts.append(full_text)
    # 简单数据增强：重复并可选地加入小扰动（此处先直接重复）
    # 如果你有更复杂的 augmentation（例如用 temperature 0.98 再生成），可以替代下面的重复逻辑
    train_texts = train_texts * 2  # 扩大数据量

    # 如果产生的 train_texts 太少（例如 <2），警告并跳过微调
    if len(train_texts) < 2:
        print("警告: 本代有效训练样本太少，跳过 SFT。本代 top_count=", len(train_texts))
    else:
        dataset = Dataset.from_dict({"text": train_texts})
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=256
            )
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        tokenized_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask'])

        # ---------- 8. SFT 微调（更保守的超参 + 梯度裁剪） ----------
        # Note: 将 fp16 关闭以避免部分环境下的 nan；如果你确认环境支持 bf16，可改回。
        sft_args = SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=1e-4,      # 降低学习率（比原先 5e-4 更稳）
            fp16=False,              # 先不使用 fp16，若你想尝试可改为 bf16=True（需硬件支持）
            bf16=False,
            logging_steps=1,
            output_dir=f"./evolve_gen_{gen}",
            save_strategy="no",
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            max_seq_length=256,
            remove_unused_columns=False,
            # TRL SFTConfig may not accept max_grad_norm; trainer arguments can set gradient clipping:
            # 我们将在 trainer.train() 前使用 model.config 或 trainer 进行裁剪（TRL 期望的字段可能不同）
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            packing=False,
            args=sft_args,
        )

        # 如果 trainer 支持设置 max_grad_norm: 尝试设置（若不支持会被忽略）
        try:
            trainer.args.max_grad_norm = 1.0
        except Exception:
            pass

        # safe training try/except to capture nan
        try:
            trainer.train()
        except Exception as e:
            print("SFT 训练异常:", e)
            print("尝试降低学习率并重试一次")
            # 简单降 lr 并重试一次
            sft_args.learning_rate = 1e-5
            trainer = SFTTrainer(
                model=model,
                train_dataset=tokenized_dataset,
                tokenizer=tokenizer,
                packing=False,
                args=sft_args,
            )
            try:
                trainer.train()
            except Exception as e2:
                print("重试仍然失败，跳过本代微调。异常：", e2)

        # 训练后释放显存并把 model 移回 cpu（你原脚本也是这样）
        model = model.cpu()
        torch.cuda.empty_cache()

# ================== 9. 保存最终模型 ==================
final_path = "EpiEvolve_TinyLlama_Final"
# 如果 model 在 CPU，把 model 移回 device 才能 save 到和 LoRA/PeFT 兼容的位置
try:
    model.to("cpu")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n5代自进化完成！最终模型已保存至: {final_path}")
except Exception as e:
    print("保存模型失败:", e)
    # 尝试通过 PEFT 保存（若适用）：
    try:
        model.save_pretrained(final_path)
    except Exception as e2:
        print("第二次保存失败:", e2)

# 结束
