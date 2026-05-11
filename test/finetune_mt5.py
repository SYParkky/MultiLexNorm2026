from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_from_disk
from huggingface_hub import login
import torch
import os

# ===== 1. Login =====
login(token="hf_")  # Replace with your token

# ===== 2. GPU Check =====
print(f"GPU:  {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# ===== 3. Config =====  
MODEL_NAME = "google/mt5-base"
MAX_LEN    = 128
CACHE_DIR  = "./data/cached_mt5"
OUTPUT_DIR = "./mt5-finetuned"
FINAL_DIR  = "./mt5-finetuned-final"

# ===== 4. Load Cached Data =====
print("\n✅ Loading cached data...")
train_tok = load_from_disk(os.path.join(CACHE_DIR, "train_tokenized"))
val_tok   = load_from_disk(os.path.join(CACHE_DIR, "val_tokenized"))
print(f"   train: {len(train_tok)} samples")
print(f"   val:   {len(val_tok)} samples")

# ===== 5. Model & Tokenizer =====
print(f"\nLoading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16
)
model = model.to("cuda")
print("Model loaded!")

# ===== 6. Training Config =====
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=24,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,

    # Training
    num_train_epochs=8,
    learning_rate=5e-4,
    bf16=True,

    # Evaluation & Saving
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=8,
    load_best_model_at_end=False,

    # Generation
    predict_with_generate=True,
    generation_max_length=MAX_LEN,

    # Logging
    logging_steps=100,
    logging_dir="./logs_mt5",
    report_to="tensorboard",

    # Windows compatibility
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
)

# ===== 7. Trainer =====
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

# ===== 8. Train =====
print("\nStarting training!")
print(f"  Model:         mT5-base (580M params)")
print(f"  Learning rate: 5e-4")

trainer.train()

# ===== 9. Save =====
print("\nSaving model...")
model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
print(f"Saved to {FINAL_DIR}")

# ===== 10. Quick Test =====
def normalize(word):
    inputs = tokenizer(word, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=32)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== Quick Test ===")
for w in ["tilfaeldigt", "u", "bcause", "lol", "ㄹㅇ"]:
    print(f"  '{w}' → '{normalize(w)}'")
