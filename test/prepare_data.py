# data/prepare_data.py
# Run once to tokenize and cache dataset
# Subsequent runs load from cache (5 seconds vs 5 minutes)

import os
from datasets import load_dataset, Dataset, load_from_disk
from huggingface_hub import login
from transformers import AutoTokenizer

# ===== Config =====
CACHE_DIR  = "./data/cached_mt5"
MODEL_NAME = "google/mt5-base"   # tokenizer is same for base/large
MAX_LEN    = 128

def flatten(split):
    """Convert sentence-level lists to word-level rows"""
    rows = []
    for example in split:
        for r, n in zip(example['raw'], example['norm']):
            rows.append({
                'raw':  r,
                'norm': n,
                'lang': example['lang']
            })
    return Dataset.from_list(rows)

def tokenize(flat_dataset, tokenizer):
    """Tokenize a flattened dataset"""
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

    return flat_dataset.map(
        preprocess,
        batched=True,
        remove_columns=['raw', 'norm', 'lang'],
    )

def get_tokenized_data(force_retokenize=False):
    """
    Load tokenized data from cache if available,
    otherwise tokenize and cache for future runs.
    """
    train_cache = os.path.join(CACHE_DIR, "train_tokenized")
    val_cache   = os.path.join(CACHE_DIR, "val_tokenized")

    # Load from cache if available
    if os.path.exists(train_cache) and not force_retokenize:
        print("✅ Loading from cache (skipping tokenization)...")
        train_tok = load_from_disk(train_cache)
        val_tok   = load_from_disk(val_cache)
        print(f"   train: {len(train_tok)} samples")
        print(f"   val:   {len(val_tok)} samples")
        return train_tok, val_tok

    # First run: tokenize and cache
    print("First run detected — tokenizing and caching...")
    print("(This takes ~5 minutes, but only happens once)")

    dataset = load_dataset("weerayut/multilexnorm2026-dev-pub")

    print("Flattening...")
    train_flat = flatten(dataset['train'])
    val_flat   = flatten(dataset['validation'])
    print(f"  train: {len(train_flat)} samples")
    print(f"  val:   {len(val_flat)} samples")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing train...")
    train_tok = tokenize(train_flat, tokenizer)

    print("Tokenizing val...")
    val_tok = tokenize(val_flat, tokenizer)

    # Save to cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    train_tok.save_to_disk(train_cache)
    val_tok.save_to_disk(val_cache)
    print(f"✅ Cached to {CACHE_DIR}")
    print("   Next run will load instantly!")

    return train_tok, val_tok


if __name__ == "__main__":
    login()  # HuggingFace login
    train_tok, val_tok = get_tokenized_data()
    print("\nDone!")
