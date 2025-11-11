# train_distill.py ← 2025 终极进化版（reward 必升！）
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
from transformers.cache_utils import DynamicCache
DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

# ================== 参数 ==================
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default='australia-covid')
parser.add_argument('--window', type=int, default=20)
parser.add_argument('--predictor_path', type=str, default='predictor_save/best_hybridgnn_predictor.pt')
parser.add_argument('--rm_path', type=str, default='rm_save/best_hybridgnn_rm.pt')
args = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# ================== 1. 加载数据 ==================
data_args = argparse.Namespace(
    dataset=args.dataset, sim_mat=f'{args.dataset}-adj', window=args.window, horizon=1,
    train=0.5, val=0.2, test=0.3, cuda=True, batch=128, is_reward_model=False,
    dropout=0.2, rnn_model='RNN', n_hidden=64, hidR=64, hidA=64, n=2, k=8, res=1
)
data_loader = DataBasicLoader(data_args)
num_regions = data_loader.m

# ================== 2. 加载模型 ==================
LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID, device_map="auto", attn_implementation="eager",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
    ), trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID, trust_remote_code=True, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
))

# ================== 3. 加载 Predictor & RM ==================
predictor = HybridGNN(argparse.Namespace(**{**vars(data_args), 'is_reward_model': False}), data_loader).to(device)
predictor.load_state_dict(torch.load(args.predictor_path, map_location=device), strict=False)
predictor.eval()

reward_model = HybridGNN(argparse.Namespace(**{**vars(data_args), 'is_reward_model': True, 'window': args.window}), data_loader).to(device)
reward_model.load_state_dict(torch.load(args.rm_path, map_location=device))
reward_model.eval()

# ================== 4. 预处理 ==================
test_X = data_loader.test[0][:32]  # 减少到 32 条
histories = test_X[:, :args.window].cpu().numpy()
prompts = [f"<|user|>\n" + "\n".join([f"Day {t+1}: {' '.join([f'{x:.6f}' for x in day])}" for t, day in enumerate(hist)]) +
           f"\nDay {args.window + 1} (predict {num_regions} regions, space separated):<|end|>\n<|assistant|>" 
           for hist in histories]
history_tensor = test_X[:, :args.window].to(device)

# ================== 5. 自进化主循环 ==================
BATCH_SIZE_GEN = 8
NUM_CANDIDATES = 4
TOP_K = 8  # 精选 Top-8
prev_best = -1e9

def parse_nums(text):
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    nums = [float(x) for x in nums]
    if len(nums) > num_regions:
        nums = nums[:num_regions]
    elif len(nums) < num_regions:
        nums.extend([0.0] * (num_regions - len(nums)))
    return nums

def normalize(pred, hist):
    p = np.array(pred)
    mean, std = np.mean(hist), np.std(hist)
    return (p - mean) / (std + 1e-8) if std > 0 else p - mean

for gen in range(1, 6):
    print(f"\n{'='*30} 第 {gen}/5 代 {'='*30}")
    model.to(device)

    # ---------- 生成（高探索）----------
    responses = []
    prompt_indices = []
    temperature = 1.0 if gen <= 2 else 0.8  # 前2代高探索
    top_p = 0.95

    for i in range(0, len(prompts), BATCH_SIZE_GEN):
        batch_prompts = prompts[i:i + BATCH_SIZE_GEN]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        for cand in range(NUM_CANDIDATES):
            # 移除 seed 锁定 → 真正随机
            outputs = model.generate(
                **inputs, max_new_tokens=256, do_sample=True,
                temperature=temperature, top_p=top_p,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            batch_resp = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            responses.extend(batch_resp)
            prompt_indices.extend(range(i, i + len(batch_prompts)))
        torch.cuda.empty_cache()

    # ---------- 解析 ----------
    answers = []
    for resp, p_idx in zip(responses, prompt_indices):
        if "<|assistant|>" in resp:
            ans = resp.split("<|assistant|>")[1].split("<|end|>")[0].strip()
        else:
            ans = resp.split("assistant>")[-1].strip()
        answers.append(ans)

    # ---------- 归一化 + Reward ----------
    llm_norm = np.stack([normalize(parse_nums(ans), histories[p_idx]) for ans, p_idx in zip(answers, prompt_indices)])
    P_llm = torch.tensor(llm_norm, dtype=torch.float32).unsqueeze(1).to(device)
    history_expanded = history_tensor[prompt_indices % len(prompts)].to(device)
    X_rm = torch.cat([history_expanded, P_llm], dim=1)

    with torch.no_grad():
        rewards = reward_model(X_rm)[0].squeeze(1).cpu().numpy()
        pred_true = predictor(history_tensor)[0].squeeze(1).cpu().numpy()
        pred_expanded = pred_true[prompt_indices % len(prompts)]

    mse = np.mean((llm_norm - pred_expanded) ** 2, axis=1)
    hybrid = 0.7 * rewards + 0.3 * (-mse)  # 更信任 RM

    # ---------- 选 Top-K ----------
    threshold = prev_best * 0.9 if gen > 1 else -1e9
    valid = hybrid > threshold
    top_idx = np.where(valid)[0][np.argsort(hybrid[valid])[-TOP_K:]] if np.any(valid) else np.argsort(hybrid)[-TOP_K:]
    best_r = hybrid[top_idx[-1]]
    prev_best = max(prev_best, best_r)
    print(f"最佳 Hybrid Reward: {best_r:.4f} ↑ | 有效样本: {sum(valid)}/{len(hybrid)}")

    # ---------- SFT（3轮）----------
    train_texts = [prompts[prompt_indices[i]] + answers[i] + "<|end|>" for i in top_idx]
    dataset = Dataset.from_dict({"text": train_texts})
    tokenized = dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=512, padding="max_length"), batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=['input_ids', 'attention_mask'])

    trainer = SFTTrainer(
        model=model, train_dataset=tokenized, tokenizer=tokenizer,
        args=SFTConfig(
            per_device_train_batch_size=1, gradient_accumulation_steps=8, num_train_epochs=3,
            learning_rate=2e-4, fp16=True, output_dir=f"./gen_{gen}", save_strategy="no",
            gradient_checkpointing=True, max_seq_length=512, remove_unused_columns=False,
        ),
    )
    trainer.train()
    model.cpu()
    torch.cuda.empty_cache()

# ================== 保存 ==================
model.save_pretrained("EpiEvolve_Final")
tokenizer.save_pretrained("EpiEvolve_Final")
print("5代自进化完成！")
