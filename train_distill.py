# train_distill.py — 自进化增强版（带变异、多样性、奖励温度）
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ATTN_IMPLEMENTATION"] = "eager"

import torch
import torch.nn.functional as F
import numpy as np
import re
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from data import DataBasicLoader
from models import HybridGNN
from transformers.cache_utils import DynamicCache
DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

# -------------------- 参数 --------------------
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default='australia-covid')
parser.add_argument('--sim_mat', type=str, default='australia-adj')
parser.add_argument('--window', type=int, default=20)
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--residual_window', type=int, default=4)
parser.add_argument('--use_residual', type=bool, default=True)
parser.add_argument('--predictor_path', type=str, default='predictor_save/best_hybridgnn_predictor.pt')
parser.add_argument('--rm_path', type=str, default='rm_save/best_hybridgnn_rm.pt')
parser.add_argument('--top_k', type=int, default=8, help="每代用于 SFT 的 top_k 候选数")
parser.add_argument('--gen_batch', type=int, default=4, help="生成时的 batch size")
parser.add_argument('--num_gen_rounds', type=int, default=2, help="每 prompt 生成几轮候选")
parser.add_argument('--max_new_tokens', type=int, default=256)
parser.add_argument('--mutation_rate', type=float, default=0.35, help="prompt 变异率")
parser.add_argument('--mutation_strength', type=float, default=0.5, help="prompt 变异强度")
parser.add_argument('--reward_temp', type=float, default=5.0, help="奖励温度放大系数")
parser.add_argument('--lambda_div', type=float, default=0.1, help="多样性惩罚系数")
parser.add_argument('--debug', action='store_true', help="开启调试打印")
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ================== 1. 加载数据 ==================
data_args = argparse.Namespace(
    dataset=args.dataset,
    sim_mat=args.sim_mat,
    window=args.window,
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

# ================== 3. 加载 Predictor ==================
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
model_dict = predictor.state_dict()
filtered_state_dict = {}
mismatched_keys = []
for k, v in state_dict.items():
    if k in model_dict and v.shape == model_dict[k].shape:
        filtered_state_dict[k] = v
    elif k in model_dict:
        mismatched_keys.append(f"{k} (Checkpoint:{v.shape} vs Model:{model_dict[k].shape})")
load_result = predictor.load_state_dict(filtered_state_dict, strict=False)
print("Predictor 智能加载成功 (非兼容权重已重新初始化)。")
if load_result.missing_keys:
    print(f"  - 缺失键: {load_result.missing_keys}")
if mismatched_keys:
    print(f"  - 维度不匹配键: {mismatched_keys}")
predictor.eval()

# ================== 4. 加载 Reward Model ==================
rm_args = argparse.Namespace(**{
    **vars(data_args),
    'is_reward_model': True,
    'window': args.window,
    'cuda': True,
    'residual_window': 0
})
reward_model = HybridGNN(rm_args, data_loader).to(device)
reward_model.load_state_dict(torch.load(args.rm_path, map_location=device))
reward_model.eval()
print("Reward Model 加载成功")

# ================== 5. 生成提示（动态 window） ==================
def create_prompt(history_window):
    lines = [f"Day {t+1}: {' '.join([f'{x:.6f}' for x in day])}"
             for t, day in enumerate(history_window)]
    history_text = '\n'.join(lines)
    return f"""<|user|>
{history_text}
Day {args.window + 1} (predict {num_regions} regions, space separated):<|end|>
<|assistant|>"""

test_X = data_loader.test[0][:16]
histories = test_X[:, :args.window].cpu().numpy()
prompts = [create_prompt(hist) for hist in histories]
print(f"生成 {len(prompts)} 个 prompts，用于自进化。")

def parse_nums(text, num_regions):
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    nums = [float(x) for x in nums]
    if len(nums) < num_regions:
        nums.extend([0.0] * (num_regions - len(nums)))
    return nums[:num_regions]

def mutate_prompt(prompt, strength=0.5):
    # 简单的随机替换数值或词语示例 — 你可以根据具体 prompt 语义设计
    parts = prompt.split('\n')
    new_parts = []
    for line in parts:
        if line.startswith("Day"):
            if np.random.rand() < strength:
                # 随机扰动一个数字
                new_line = re.sub(r"([-+]?\d*\.\d+)", lambda m: f"{float(m.group(1))*(1 + (np.random.rand()-0.5)*0.2):.6f}", line)
                new_parts.append(new_line)
            else:
                new_parts.append(line)
        else:
            new_parts.append(line)
    return '\n'.join(new_parts)

# ================== 6. 自进化主循环（增强版） ==================
BATCH_SIZE_GEN = args.gen_batch
MAX_NEW_TOKENS = args.max_new_tokens
NUM_GEN_ROUNDS = args.num_gen_rounds
TOP_K = args.top_k

for gen in range(1, 6):
    print(f"\n{'='*20} 第 {gen}/5 代自进化 {'='*20}")
    model = model.to(device)
    model.eval()

    # 1. 生成候选 prompts (变异 + 多轮生成)
candidate_prompts = []
base_indices = []  # 每个候选对应哪个原始 prompt
for base_idx, p in enumerate(prompts):
    # 原始 prompt
    candidate_prompts.append(p)
    base_indices.append(base_idx)
    # 变异 prompt
    if np.random.rand() < args.mutation_rate:
        mutated = mutate_prompt(p, strength=args.mutation_strength)
        candidate_prompts.append(mutated)
        base_indices.append(base_idx)

# 2. 对每个候选 prompt 生成预测
answers = []
prompt_indices = []  # 对应 base_indices
for i in range(0, len(candidate_prompts), BATCH_SIZE_GEN):
    batch_prompts = candidate_prompts[i:i + BATCH_SIZE_GEN]
    batch_base = base_indices[i:i + BATCH_SIZE_GEN]
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)
    input_lens = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1).tolist()

    for r in range(NUM_GEN_ROUNDS):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.9 if r>0 else 0.7,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
        for out_ids, in_len, base_idx in zip(outputs, input_lens, batch_base):
            new_text = tokenizer.decode(out_ids[in_len:], skip_special_tokens=True).strip()
            answers.append(new_text)
            prompt_indices.append(base_idx)  # 注意这里是 base_idx
    torch.cuda.empty_cache()

    # ✅ 检查防止越界
    prompt_indices = [min(idx, len(test_X)-1) for idx in prompt_indices]


    # 3. 解析答案为数值 + 拼接 RM 输入 + 计算 reward
    parsed = [parse_nums(ans, num_regions) for ans in answers]
    P_llm_raw = torch.tensor(parsed, dtype=torch.float32).unsqueeze(1).to(device)

    # 尺度映射
    ref_last_hist = test_X[prompt_indices, args.window-1].to(device)
    eps = 1e-6
    scale = ref_last_hist.abs() + eps
    p_raw = P_llm_raw.squeeze(1)
    p_min = p_raw.min(dim=1, keepdim=True).values
    p_max = p_raw.max(dim=1, keepdim=True).values
    denom = (p_max - p_min).clamp(min=1e-9)
    p_norm = (p_raw - p_min) / denom
    P_llm_scaled = (p_norm * scale).unsqueeze(1)

    # 构造 RM 输入并评分
    rewards = []
    prompt_embeds = []
    for idx, pidx in enumerate(prompt_indices):
        hist = test_X[pidx, :args.window].to(device)
        inp = torch.cat([hist.unsqueeze(0), P_llm_scaled[idx:idx+1]], dim=1)  # shape (1, window+1, num_regions)
        with torch.no_grad():
            r_val, _ = reward_model(inp)
        rewards.append(r_val.item())
        # 用模型编码 prompt 为 embedding（示例用 LLM encoder）
        with torch.no_grad():
            emb = model.encode(candidate_prompts[pidx], convert_to_tensor=True)
        prompt_embeds.append(emb.cpu())
    rewards = torch.tensor(rewards, dtype=torch.float32)
    prompt_embeds = torch.stack(prompt_embeds)

    # 4. 奖励温度 + 多样性惩罚
    rewards_scaled = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    rewards_scaled = torch.sigmoid(rewards_scaled * args.reward_temp)
    adjusted_rewards = rewards_scaled.clone()
    for i in range(len(adjusted_rewards)):
        sim = cosine_similarity(prompt_embeds[i:i+1], prompt_embeds).mean()
        adjusted_rewards[i] -= args.lambda_div * sim

    # 5. 选 Top-K
    K = min(TOP_K, len(adjusted_rewards))
    topk_indices = torch.topk(adjusted_rewards, k=K).indices.tolist()
    top_prompts = [candidate_prompts[idx] for idx in topk_indices]
    top_rewards = adjusted_rewards[topk_indices].cpu().numpy()

    print("本代选出的 Top 候选（增强版）:")
    for rank, idx in enumerate(topk_indices, 1):
        print(f"  Rank {rank}: idx={idx}, reward={top_rewards[rank-1]:.6f}")
        if args.debug:
            print("    prompt:", candidate_prompts[idx][:120].replace('\n',' '))

    print(f">> rewards 分布: mean={adjusted_rewards.mean().item():.6f}, std={adjusted_rewards.std().item():.6f}, max={adjusted_rewards.max().item():.6f}")

    # 6. SFT 微调（用 Top-K prompts）
    train_texts = [top_prompts[i] + "\n" + answers[topk_indices[i]] + "<|end|>" for i in range(len(top_prompts))]
    train_texts = train_texts * 2  # 简单重复扩充

    if len(train_texts) < 2:
        print("警告：本代训练样本太少，跳过 SFT。")
    else:
        dataset = Dataset.from_dict({"text": train_texts})
        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=256
            )
        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        tokenized.set_format(type="torch", columns=['input_ids', 'attention_mask'])

        sft_args = SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=1e-4,
            fp16=False,
            bf16=False,
            logging_steps=1,
            output_dir=f"./evolve_gen_{gen}",
            save_strategy="no",
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            max_seq_length=256,
            remove_unused_columns=False,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=tokenized,
            tokenizer=tokenizer,
            packing=False,
            args=sft_args,
        )
        try:
            trainer.args.max_grad_norm = 1.0
        except Exception:
            pass

        try:
            trainer.train()
        except Exception as e:
            print("SFT训练异常，重试 with lower lr")
            sft_args.learning_rate = 1e-5
            trainer = SFTTrainer(
                model=model,
                train_dataset=tokenized,
                tokenizer=tokenizer,
                packing=False,
                args=sft_args,
            )
            try:
                trainer.train()
            except Exception as e2:
                print("重试失败，跳过本代微调。", e2)

        model = model.cpu()
        torch.cuda.empty_cache()

# ================== 保存最终模型 ==================
final_path = "EpiEvolve_TinyLlama_Final"
try:
    model.to("cpu")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n5代自进化完成！最终模型已保存至: {final_path}")
except Exception as e:
    print("保存模型失败:", e)
    try:
        model.save_pretrained(final_path)
    except Exception as e2:
        print("再次保存失败:", e2)

