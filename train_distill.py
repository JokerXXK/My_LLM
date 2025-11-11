# train_distill.py ← 2025 终极信号强化版 (已应用所有修复点)
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
test_X = data_loader.test[0][:64]
histories = test_X[:, :args.window].cpu().numpy()
prompts = [f"<|user|>\n" + "\n".join([f"Day {t+1}: {' '.join([f'{x:.6f}' for x in day])}" for t, day in enumerate(hist)]) +
           f"\nDay {args.window + 1} (predict {num_regions} regions, space separated):<|end|>\n<|assistant|>" 
           for hist in histories]

history_tensor = test_X[:, :args.window].to(device) # (64, window, R)

# ================== 5. 自进化主循环 ==================
BATCH_SIZE_GEN = 8
NUM_CANDIDATES = 3 # 每条 prompt 生成 3 个候选
TOP_K = 8 # <-- 修正点 4：降低到 8，精选优质样本，强化信号
prev_best = -1e9

def parse_nums(text):
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    parsed_nums = [float(x) for x in nums] 
    
    padding_count = max(0, num_regions - len(parsed_nums))
    return (parsed_nums + [0.0] * padding_count)[:num_regions]

def normalize(pred, hist):
    p = np.array(pred)
    std = np.std(hist)
    return (p - np.mean(hist)) / (std + 1e-8)

for gen in range(1, 6):
    print(f"\n{'='*25} 第 {gen}/5 代 {'='*25}")
    model.to(device)

    # ---------- 1. 多候选生成（强制多样性）----------
    responses = []
    prompt_indices = []
    
    # <-- 修正点 5：平衡探索与利用
    temperature = 1.0 if gen <= 3 else 0.8 # Gen 1-3 探索 (1.0)，Gen 4-5 利用 (0.8)
    top_p = 0.95

    for i in range(0, len(prompts), BATCH_SIZE_GEN):
        batch_prompts = prompts[i:i + BATCH_SIZE_GEN]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        for cand in range(NUM_CANDIDATES):
            # 移除 torch.manual_seed，实现真正的随机多样性
            # torch.manual_seed(42 + gen * 100 + cand) # <-- 修正点 2：移除
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, # <-- 修正点 1：恢复到 256
                do_sample=True,
                temperature=temperature, 
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            batch_resp = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            responses.extend(batch_resp)
            prompt_indices.extend(range(i, i + len(batch_prompts)))
        torch.cuda.empty_cache()

    # ---------- 2. 解析 ----------
    # ... (解析逻辑不变)
    answers = []
    for resp, p_idx in zip(responses, prompt_indices):
        if "<|assistant|>" in resp:
            ans = resp.split("<|assistant|>")[1].split("<|end|>")[0].strip()
        else:
            ans = resp.replace(prompts[p_idx], "").strip()
        answers.append(ans)

    # ---------- 3. 归一化 + Reward ----------
    # ... (奖励计算逻辑不变)
    llm_norm = np.stack([normalize(parse_nums(ans), histories[p_idx]) for ans, p_idx in zip(answers, prompt_indices)])
    P_llm = torch.tensor(llm_norm, dtype=torch.float32).unsqueeze(1).to(device)

    indices_numpy = np.array(prompt_indices)
    indices_tensor = torch.from_numpy(indices_numpy).to(device)
    history_expanded = history_tensor[indices_tensor]
    X_rm = torch.cat([history_expanded, P_llm], dim=1)

    with torch.no_grad():
        rewards = reward_model(X_rm)[0].squeeze(1).cpu().numpy()
        pred_true = predictor(history_tensor)[0].squeeze(1).cpu().numpy()
        pred_expanded = pred_true[indices_numpy]

    mse = np.mean((llm_norm - pred_expanded) ** 2, axis=1)
    hybrid = 0.6 * rewards + 0.4 * (-mse)

    # ---------- 4. 选 Top-K ----------
    threshold = prev_best * 0.9 if gen > 1 else -1e9
    valid = hybrid > threshold
    
    if np.any(valid):
        top_idx = np.where(valid)[0][np.argsort(hybrid[valid])[-TOP_K:]]
    else:
        top_idx = np.argsort(hybrid)[-TOP_K:]

    best_r = hybrid[top_idx[-1]]
    prev_best = max(prev_best, best_r)
    print(f"最佳 Hybrid Reward: {best_r:.4f} ↑ | 样本: {len(top_idx)}/{len(hybrid)}")

    # ---------- 5. SFT ----------
    train_texts = [prompts[prompt_indices[i]] + answers[i] + "<|end|>" for i in top_idx]
    dataset = Dataset.from_dict({"text": train_texts})
    tokenized = dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=512, padding="max_length"), batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=['input_ids', 'attention_mask'])

    trainer = SFTTrainer(
        model=model, 
        train_dataset=tokenized,
        args=SFTConfig(
            per_device_train_batch_size=1, gradient_accumulation_steps=4, 
            num_train_epochs=3, # <-- 修正点 3：恢复到 3 个 Epoch
            learning_rate=2e-4, # <-- 修正点 3：提高学习率到 2e-4
            fp16=True, output_dir=f"./gen_{gen}", save_strategy="no",
            gradient_checkpointing=True, 
            remove_unused_columns=False,
        ),
    )
    trainer.train()
    model.cpu()
    torch.cuda.empty_cache()

# ================== 保存 ==================
model.save_pretrained("EpiEvolve_Final")
tokenizer.save_pretrained("EpiEvolve_Final")
print("完成！")
