from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, Dataset, load_from_disk
from huggingface_hub import login
import torch
import os

# ===== 1. Login =====
login(token="hf_")  # Replace with your token

# ===== 2. GPU Check =====
print(f"GPU:  {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# ===== 3. Config =====
MODEL_NAME     = "google/byt5-large"
MAX_LEN        = 128
CACHE_DIR      = "./data/cached"
OUTPUT_DIR     = "./byt5-large-finetuned-0511"
FINAL_DIR      = "./byt5-large-finetuned-0511-final"

# Resume from checkpoint (set to None to start fresh)
# Example: "./byt5-large-finetuned/checkpoint-20031"
RESUME_FROM    = None

# ===== 4. Tokenization Cache =====
def flatten(split):
    rows = []
    for example in split:
        for r, n in zip(example['raw'], example['norm']):
            rows.append({
                'raw':  r,
                'norm': n,
                'lang': example['lang']
            })
    return Dataset.from_list(rows)

def get_tokenized_data(tokenizer, force_retokenize=False):
    train_cache = os.path.join(CACHE_DIR, "train_tokenized")
    val_cache   = os.path.join(CACHE_DIR, "val_tokenized")

    # Load from cache if available
    if os.path.exists(train_cache) and not force_retokenize:
        print("✅ Loading from cache (tokenization skipped)...")
        train_tok = load_from_disk(train_cache)
        val_tok   = load_from_disk(val_cache)
        print(f"   train: {len(train_tok)} samples")
        print(f"   val:   {len(val_tok)} samples")
        return train_tok, val_tok

    # First run: tokenize and cache
    print("Tokenizing for first time (~5 min)...")
    dataset = load_dataset("weerayut/multilexnorm2026-dev-pub")

    train_flat = flatten(dataset['train'])
    val_flat   = flatten(dataset['validation'])
    print(f"  train: {len(train_flat)} samples")
    print(f"  val:   {len(val_flat)} samples")

    def preprocess(examples):
        inputs = tokenizer(
            examples['raw'],
            max_length=MAX_LEN,
            truncation=True,
            padding=False
        )
        labels = tokenizer(
            examples['norm'],
            max_length=MAX_LEN,
            truncation=True,
            padding=False
        )
        inputs['labels'] = labels['input_ids']
        return inputs

    print("  Tokenizing train...")
    train_tok = train_flat.map(
        preprocess,
        batched=True,
        remove_columns=['raw', 'norm', 'lang'],
    )
    print("  Tokenizing val...")
    val_tok = val_flat.map(
        preprocess,
        batched=True,
        remove_columns=['raw', 'norm', 'lang'],
    )

    # Save cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    train_tok.save_to_disk(train_cache)
    val_tok.save_to_disk(val_cache)
    print(f"✅ Cached to {CACHE_DIR} (next run will be instant!)")

    return train_tok, val_tok

# ===== 5. Model & Tokenizer =====
print(f"\nLoading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16
)
model = model.to("cuda")
print("Model loaded!")

# ===== 6. Load Data (from cache) =====
train_tok, val_tok = get_tokenized_data(tokenizer)

# ===== 7. Training Config =====
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    # RTX 5080 (17GB) optimized for ByT5-large
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,      # effective batch = 32
    gradient_checkpointing=True,        # essential for large on 17GB

    # Training
    num_train_epochs=4,                 # 1 epoch done → 3 more = 4 total
    learning_rate=3e-4,
    bf16=True,

    # Evaluation & Saving
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=8,
    load_best_model_at_end=True,

    # Generation
    predict_with_generate=True,
    generation_max_length=MAX_LEN,

    # Logging
    logging_steps=100,
    logging_dir="./logs_large",
    report_to="tensorboard",

    # Windows compatibility
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
)

# ===== 8. Trainer =====
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    ),
)

# ===== 9. Train =====
print("\nStarting training!")
print(f"  Model:          ByT5-large (1.2B params)")
print(f"  Resume from:    {RESUME_FROM}")

# Resume from checkpoint if specified
if RESUME_FROM and os.path.exists(RESUME_FROM):
    print(f"\n✅ Resuming from: {RESUME_FROM}")
    trainer.train(resume_from_checkpoint=RESUME_FROM)
else:
    print("\n⚠️  No checkpoint found, starting fresh")
    trainer.train()

# ===== 10. Save =====
print("\nSaving model...")
model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
print(f"Saved to {FINAL_DIR}")

# ===== 11. Quick Test =====
def normalize(word):
    inputs = tokenizer(word, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=32)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== Quick Test ===")
for w in ["tilfaeldigt", "u", "bcause", "lol", "ㄹㅇ"]:
    print(f"  '{w}' → '{normalize(w)}'")
