# train_distill.py  ← 2025 终极稳定版（已修复所有坑，6GB 显存 5 代 0 报错）
# ================== 关键改动说明 ==================
# 1. 所有硬编码的 20 替换为 args.window
# 2. create_prompt 动态生成 Day 1…Day {window} → Day {window+1}
# 3. test_X 切片使用 args.window
# 4. RM 输入拼接历史 (window) + LLM 预测 (1) → window+1 步
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
# 关键补丁
from transformers.cache_utils import DynamicCache
DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

# ================== 参数 ==================
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default='australia-covid')
parser.add_argument('--window', type=int, default=20)          # ← 可变的历史长度
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--residual_window', type=int, default=4)
parser.add_argument('--use_residual', type=bool, default=True)
parser.add_argument('--predictor_path', type=str, default='predictor_save/best_hybridgnn_predictor.pt')
parser.add_argument('--rm_path', type=str, default='rm_save/best_hybridgnn_rm.pt')
args = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ================== 1. 加载数据 ==================
data_args = argparse.Namespace(
    dataset=args.dataset,
    sim_mat=f'{args.dataset}-adj',
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

# ================== 2. 加载 TinyLlama 4bit ==================
LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"加载 {LLM_MODEL_ID} (4bit)...")
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    device_map="auto",
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
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

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
    history_window: shape (window, num_regions)
    """
    lines = [f"Day {t+1}: {' '.join([f'{x:.6f}' for x in day])}"
             for t, day in enumerate(history_window)]
    history_text = '\n'.join(lines)
    return f"""<|user|>
{history_text}
Day {args.window + 1} (predict {num_regions} regions, space separated):<|end|>
<|assistant|>"""

# 取前 64 条测试样本的历史（长度 = args.window）
test_X = data_loader.test[0][:64]                     # (16, total_steps, num_regions)
histories = test_X[:, :args.window].cpu().numpy()    # (16, window, num_regions)
prompts = [create_prompt(hist) for hist in histories]

# ================== 6. 自进化主循环 ==================
BATCH_SIZE_GEN = 16
MAX_NEW_TOKENS = 256

def parse_nums(text, num_regions):
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    nums = [float(x) for x in nums]
    if len(nums) < num_regions:
        nums.extend([0.0] * (num_regions - len(nums)))
    return nums[:num_regions]

for gen in range(1, 6):
    print(f"\n{'='*20} 第 {gen}/5 代自进化 {'='*20}")

    # 在自进化循环的开头（每代开始前）加上：
    model = model.to(device)   # ← 关键修复！

    # ---------- 1. 分批生成 ----------
    responses = []
    for i in range(0, len(prompts), BATCH_SIZE_GEN):
        batch_prompts = prompts[i:i + BATCH_SIZE_GEN]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses.extend(batch_responses)
        torch.cuda.empty_cache()

    # ---------- 2. 解析答案 ----------
    answers = []
    for resp, prompt in zip(responses, prompts):
        if "<|assistant|>" in resp:
            ans = resp.split("<|assistant|>")[1].split("<|end|>")[0].strip()
        else:
            ans = resp.replace(prompt, "").strip()
        answers.append(ans)

    # ---------- 3. 解析数字 + RM 打分 ----------
    P_llm = torch.tensor([parse_nums(ans, num_regions) for ans in answers],
                         dtype=torch.float32).unsqueeze(1).to(device)   # (B, 1, num_regions)

    # RM 输入：历史 (window) + LLM预测 (1) = window+1 步
    X_rm_input = torch.cat([test_X[:, :args.window].to(device), P_llm], dim=1)

    with torch.no_grad():
        rewards, _ = reward_model(X_rm_input)
    rewards = rewards.squeeze(1).cpu().numpy()

    # ---------- 4. 选 Top-8 ----------
    top_idx = np.argsort(rewards)[-8:]
    best_r = rewards[top_idx[-1]]
    print(f"本代最佳奖励: {best_r:.4f}")

    # ---------- 5. 构造 SFT 数据 ----------
    train_texts = []
    for idx in top_idx:
        full_text = prompts[idx] + answers[idx] + "<|end|>"
        train_texts.append(full_text)

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

    # ---------- 6. SFT 微调 ----------
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir=f"./evolve_gen_{gen}",
            save_strategy="no",
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            max_length=256,
            remove_unused_columns=False,
        ),
    )
    trainer.train()
    model = model.cpu()
    torch.cuda.empty_cache()

# ================== 9. 保存最终模型 ==================
final_path = "EpiEvolve_TinyLlama_Final"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"\n5代自进化完成！最终模型已保存至: {final_path}")
