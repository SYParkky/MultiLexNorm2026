import json
import random
import os

INPUT_FILE  = "data/th_train.jsonl"  
OUTPUT_FILE = "data/th_train_augmented.jsonl"
RATIO       = 0.5  
SEED        = 42

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def create_identity_pairs(data, ratio=0.5):
    """
    For each sentence, collect tokens that are already standard
    (raw == norm) and bundle them into a new identity sentence.

    Args:
        data  : list of dicts with keys 'raw', 'norm', 'lang'
        ratio : how many identity sentences to add relative to
                original dataset size (0.5 = half as many)
    Returns:
        list of identity pair dicts
    """
    identity_sentences = []

    for item in data:
        raw_tokens  = item['raw']
        norm_tokens = item['norm']

        standard_words = [
            r for r, n in zip(raw_tokens, norm_tokens) if r == n
        ]

        if len(standard_words) == 0:
            continue

        identity_sentences.append({
            'raw':  standard_words,
            'norm': standard_words,
            'lang': 'th'
        })

    target_count = int(len(data) * ratio)
    if len(identity_sentences) > target_count:
        identity_sentences = random.sample(identity_sentences, target_count)

    return identity_sentences


def augment_thai(data, ratio=0.5, seed=42):
    """
    Merge original data with generated identity pairs and shuffle.

    Args:
        data  : list of dicts (original Thai training data)
        ratio : identity pair ratio relative to original size
        seed  : random seed for reproducibility
    Returns:
        augmented list of dicts
    """
    random.seed(seed)

    identity_pairs = create_identity_pairs(data, ratio=ratio)
    augmented      = data + identity_pairs
    random.shuffle(augmented)

    print(f"[Thai Augmentation]")
    print(f"  Original sentences  : {len(data):,}")
    print(f"  Identity pairs added: {len(identity_pairs):,}")
    print(f"  Total               : {len(augmented):,}")
    print(f"  Ratio               : 1:{round(len(data)/max(len(identity_pairs),1), 1)}")
    return augmented

def save_jsonl(data, filepath):
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Saved → {filepath}")


def verify(filepath, n=5):
    print(f"\n[Verification] First {n} examples from {filepath}:")
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            tag  = "IDENTITY" if item['raw'] == item['norm'] else "NORMAL "
            print(f"  [{tag}] raw : {item['raw'][:6]}")
            print(f"          norm: {item['norm'][:6]}")
            print()
            if i >= n - 1:
                break

    total = identity = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            total += 1
            if item['raw'] == item['norm']:
                identity += 1
    normal = total - identity
    print(f"  Total   : {total:,}")
    print(f"  Normal  : {normal:,}  ({100*normal/total:.1f}%)")
    print(f"  Identity: {identity:,}  ({100*identity/total:.1f}%)")

if __name__ == "__main__":
    print("=" * 50)
    print("Thai Identity Pair Augmentation")
    print("=" * 50)

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Make sure you are running this from the repo root directory.")
        exit(1)

    print(f"\nLoading {INPUT_FILE} ...")
    data = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(data):,} sentences.")

    print()
    augmented = augment_thai(data, ratio=RATIO, seed=SEED)

    print()
    save_jsonl(augmented, OUTPUT_FILE)

    verify(OUTPUT_FILE, n=5)

    print("\nDone. Send th_train_augmented.jsonl to the training pipeline.")
