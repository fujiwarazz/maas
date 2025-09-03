import os, re, torch, numpy as np
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

POLICY_INIT = "/data/hxzh/unsloth/sft/qwen3_sft_merged_16bit"   
REF_INIT    = "/data/hxzh/unsloth/sft/qwen3_sft_merged_16bit"   
DATA_JSON   = "/data/hxzh/LLaMA-Factory/data/infinity_preference_custom_1500_sft.json"
OUT_DIR     = "outputs_grpo_qwen32b_cy_single"
MAX_SEQ_LEN = 2048
MAX_PROMPT  = 768
LORA_R      = 8

print("Visible GPUs:", os.environ.get("CUDA_VISIBLE_DEVICES"))

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=POLICY_INIT,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    dtype=torch.bfloat16,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=LORA_R,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

gen = model.generation_config
gen.do_sample = True
gen.temperature = 0.8
gen.top_p = 0.9
gen.top_k = 50
gen.repetition_penalty = 1.05
model.generation_config = gen

ref_model, _ = FastLanguageModel.from_pretrained(
    model_name=REF_INIT,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)
def parse_instruction(instruction: str):
    sys_prompt, user_prompt = "", instruction
    if "[SYSTEM]" in instruction and "[INSTRUCTION]" in instruction:
        parts = instruction.split("[INSTRUCTION]", 1)
        sys_prompt = parts[0].replace("[SYSTEM]", "").strip()
        user_prompt = parts[1].strip()
    return sys_prompt, user_prompt

def get_rl_dataset(file_path: str, tokenizer) -> Dataset:
    data = load_dataset("json", data_files={"train": file_path})["train"]
    def transform(example):
        sys_prompt, user_text = parse_instruction(example["instruction"])
        full_user = user_text + ("\n\n" + example["input"] if example.get("input") else "")
        msgs = [
            {"role": "system", "content": sys_prompt + "\n(Your final visible answer MUST be wrapped by <cy_start> ... <cy_end>.)"},
            {"role": "user", "content": full_user},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt}
    return data.map(transform, remove_columns=list(data.features))

dataset = get_rl_dataset(DATA_JSON, tokenizer)


THINK_CLOSE_RE = re.compile(r"</think>", flags=re.IGNORECASE)
def after_think(text: str) -> str:
    parts = THINK_CLOSE_RE.split(text)
    return parts[-1].strip()

START = "<cy_start>"
END   = "<cy_end>"

def _flatten(xs):
    return [y for x in xs for y in (x if isinstance(x, list) else [x])]

def _prefix_match_ratio(s, prefix):
    n = min(len(s), len(prefix))
    i = 0
    while i < n and s[i] == prefix[i]:
        i += 1
    return i / len(prefix)

def _suffix_match_ratio(s, suffix):
    n = min(len(s), len(suffix))
    i = 0
    while i < n and s[-1-i] == suffix[-1-i]:
        i += 1
    return i / len(suffix)

def cy_format_reward(completions, **kwargs):
    completions = _flatten(completions)
    max_len =  MAX_SEQ_LEN - MAX_PROMPT
    out = []
    for c in completions:
        v = after_think(c).strip()

        start_ratio = _prefix_match_ratio(v.lstrip(), START)
        end_ratio   = _suffix_match_ratio(v.rstrip(), END)

        start_ok = v.lstrip().startswith(START)
        end_ok   = v.rstrip().endswith(END)

        inner = v
        if start_ok: inner = inner[len(START):]
        if end_ok:   inner = inner[:-len(END)]
        inner_ok = len(inner.strip()) >= 10

        pre = v.split(START, 1)[0]
        post = v[v.rfind(END)+len(END):] if end_ok else ""

        s = 0.35 * start_ratio + 0.35 * end_ratio \
            + (0.2 if inner_ok else 0.0) \
            - (0.15 if pre.strip() else 0.0) \
            - (0.15 if post.strip() else 0.0)

        if len(c) >= max_len - 5:  # 如想精确，可改为 token 计数
            s -= 0.6

        s += float(np.random.normal(scale=1e-3))
        out.append(float(s))
    return out

def soft_length_reward(completions, **kwargs):
    completions = _flatten(completions)
    max_len =  MAX_SEQ_LEN - MAX_PROMPT
    target = 1000
    sigma  = 250.0
    out = []
    for c in completions:
        v = after_think(c)
        L = len(v)
        s = float(np.exp(-0.5 * ((L - target) / sigma) ** 2))
        if L >= max_len - 5:
            s -= 0.3
        out.append(s)
    return out


args = GRPOConfig(
    beta=0.05,
    learning_rate=2e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    num_generations=2,                        
    max_prompt_length=MAX_PROMPT,
    max_completion_length=MAX_SEQ_LEN - MAX_PROMPT,  
    scale_rewards=False,
    output_dir=OUT_DIR,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    optim="adamw_8bit",                      
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    seed=3407,
    bf16=True,
)

trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,
    args=args,
    train_dataset=dataset,
    reward_funcs=[cy_format_reward],
)

trainer.train()
trainer.save_model(OUT_DIR)  # 保存 LoRA 适配器
print("GRPO done. LoRA saved to:", OUT_DIR)

model.save_pretrained_merged(OUT_DIR + "_merged", tokenizer, save_method="merged_16bit")
print("Merged policy saved to:", OUT_DIR + "_merged")