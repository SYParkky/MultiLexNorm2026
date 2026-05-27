import json
import random
import os

INPUT_FILE       = "data/ja_train.jsonl"  
OUTPUT_FILE      = "data/ja_train_augmented.jsonl"
AUGMENT_FACTOR   = 3    
SEED             = 42


def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


ENDING_MAP = {
    'です':     'っす',
    'ます':     'っす',
    'ている':   'てる',
    'ていた':   'てた',
    'している': 'してる',
    'ません':   'ません',
    'でした':   'でした',
}


def corrupt_word(word):
    """
    Apply one random informal Japanese writing rule to a single token.
    Returns the corrupted form, or the original if no rule fires.

    Each rule has an independent probability so the overall corruption
    rate stays realistic and not every word gets changed.
    """
    if not word or len(word) < 1:
        return word

    roll = random.random()

    if roll < 0.15:
        w_count = random.randint(1, 4)
        return word + ('w' * w_count)

    elif roll < 0.25:
        tilde_count = random.randint(1, 3)
        return word + ('～' * tilde_count)

    elif roll < 0.33 and len(word) > 1:
        return word + word[-1]

    elif roll < 0.53:
        for formal, informal in ENDING_MAP.items():
            if word.endswith(formal):
                return word[:-len(formal)] + informal
      
    elif roll < 0.61 and len(word) > 2:
        pos = len(word) - 1
        return word[:pos] + 'っ' + word[pos:]

    return word  

def corrupt_sentence(norm_tokens):
    """
    Apply word-level corruption to a list of tokens.
    Returns a new list where some tokens may be informally written.
    norm_tokens is never modified.
    """
    return [corrupt_word(w) for w in norm_tokens]

def augment_japanese(data, augment_factor=3, seed=42):
    """
    For each sentence in the original data, generate augment_factor
    noisy variants. Only 'raw' changes — 'norm' stays fixed.

    Args:
        data           : list of dicts with keys 'raw', 'norm', 'lang'
        augment_factor : number of synthetic variants per original
        seed           : random seed for reproducibility
    Returns:
        augmented list of dicts (originals + synthetic pairs, shuffled)
    """
    random.seed(seed)
    augmented = list(data)  

    added = 0
    for item in data:
        norm_tokens = list(item['norm'])  

        for _ in range(augment_factor):
            noisy_tokens = corrupt_sentence(norm_tokens)

            if noisy_tokens != norm_tokens:
                augmented.append({
                    'raw':  noisy_tokens,   
                    'norm': norm_tokens,   
                    'lang': 'ja'
                })
                added += 1

    random.shuffle(augmented)

    print(f"[Japanese Augmentation]")
    print(f"  Original sentences     : {len(data):,}")
    print(f"  Synthetic pairs added  : {added:,}")
    print(f"  Total                  : {len(augmented):,}")
    print(f"  Augment factor used    : {augment_factor}x")
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
            item    = json.loads(line)
            changed = item['raw'] != item['norm']
            tag     = "SYNTHETIC" if changed else "ORIGINAL "
            print(f"  [{tag}] raw : {item['raw'][:6]}")
            print(f"           norm: {item['norm'][:6]}")
            print()
            if i >= n - 1:
                break

    total = synthetic = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            total += 1
            if item['raw'] != item['norm']:
                synthetic += 1
    original = total - synthetic
    print(f"  Total     : {total:,}")
    print(f"  Original  : {original:,}  ({100*original/total:.1f}%)")
    print(f"  Synthetic : {synthetic:,}  ({100*synthetic/total:.1f}%)")

    print(f"\n[Rule Hit Examples]")
    examples = {'www': [], '～': [], 'duplication': [], 'ending': [], 'っ': []}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            for r, n in zip(item['raw'], item['norm']):
                if r == n:
                    continue
                if 'w' in r and r.endswith('w'):
                    examples['www'].append((r, n))
                elif '～' in r:
                    examples['～'].append((r, n))
                elif 'っ' in r and 'っ' not in n:
                    examples['っ'].append((r, n))
                elif len(r) > len(n) and r[:-1] == n:
                    examples['duplication'].append((r, n))
                elif r != n:
                    examples['ending'].append((r, n))

    for rule, exs in examples.items():
        if exs:
            sample = exs[:2]
            print(f"  Rule [{rule}]: {sample}")

if __name__ == "__main__":
    print("=" * 50)
    print("Japanese Rule-Based Augmentation")
    print("=" * 50)

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Make sure you are running this from the repo root directory.")
        exit(1)

    print(f"\nLoading {INPUT_FILE} ...")
    data = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(data):,} sentences.")

    print()
    augmented = augment_japanese(data, augment_factor=AUGMENT_FACTOR, seed=SEED)

    print()
    save_jsonl(augmented, OUTPUT_FILE)

    verify(OUTPUT_FILE, n=5)

    print("\nDone. Send ja_train_augmented.jsonl to the training pipeline.")
