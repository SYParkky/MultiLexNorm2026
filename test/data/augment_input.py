"""
data/augment_input.py — Input-Variation Augmentation for MultiLexNorm2026

Covers languages with low ERR where more training variety is needed.
Strategy: keep 'norm' (output) fixed, only vary 'raw' (input).
This guarantees zero label noise — the ground truth never changes.

Per-language strategies based on error analysis:
    th (Thai)     ERR 10.81  FP=404  -> identity pair injection + SNS corruptions
    ja (Japanese) ERR 15.23          -> rule-based SNS corruption
    ko (Korean)   ERR  3.61          -> consonant/filler variation
    en (English)  ERR 17.06          -> contraction + vowel repetition
    de (German)   ERR  9.74          -> umlaut removal + casing

Thai FP analysis:
    FP=404 means the model over-normalizes standard Thai it should leave alone.
    Fix: (1) whole-sentence identity injection to teach "leave standard text alone",
         (2) rule-based SNS corruptions so the model also learns what real slang
             looks like, preventing it from treating standard text as slang.

Usage:
    python data/augment_input.py
    python data/augment_input.py --langs th ja --output data/input_aug.jsonl
"""

import json, random, os, argparse

DEFAULT_LANGS  = ["th", "ja", "ko", "en", "de"]
DEFAULT_OUTPUT = "data/input_aug.jsonl"
DEFAULT_FACTOR = 3
THAI_RATIO     = 0.5   # identity sentences as fraction of original data
THAI_CORRUPT_FACTOR = 2  # rule-corrupt passes per original sentence
SEED           = 42

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: data.append(json.loads(line))
    return data

def save_jsonl(data, filepath):
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data):,} examples -> {filepath}")

def load_lang_data(lang):
    for path in [f"data/{lang}_train.jsonl", f"data/cached/{lang}_train.jsonl"]:
        if os.path.exists(path):
            return load_jsonl(path)
    print(f"  [WARNING] No training file found for '{lang}'")
    return []

# ---------------------------------------------------------------------------
# Thai augmentation
# ---------------------------------------------------------------------------

# Informal particle map: norm form -> list of SNS surface variants.
# Used to synthesise raw inputs the model must learn to normalise.
THAI_PARTICLE_VARIANTS = {
    "ครับ":    ["ค้าบ", "ครับบ", "คับ", "คร้าบ"],
    "ค่ะ":     ["ค่า", "ค้า", "ค่ะๆ"],
    "นะ":      ["นะๆ", "เนอะ", "นะเนอะ"],
    "ไง":      ["ไงล่ะ", "ไงอ่ะ"],
    "อะ":      ["อ่ะ", "อ่ะๆ"],
    "ก็":      ["ก้อ"],
    "แล้ว":    ["แระ", "แล้วอ่ะ"],
    "อย่างไร": ["ยังไง", "ยังไงอ่ะ"],
    "ทำไม":   ["ทำไมอ่ะ", "ทำไมล่ะ"],
}

# For augmentation we synthesise them from the norm side.
THAI_LAUGH_TOKENS = ["555", "5555", "55555", "555555"]

def _is_thai_char(c):
    return "\u0e00" <= c <= "\u0e7f"

def _repeat_final_char(word, lo=1, hi=4):
    """กกกกก-style emphasis: repeat the final character."""
    if not word:
        return word
    last = word[-1]
    if _is_thai_char(last):
        return word + last * random.randint(lo, hi)
    return word

def _repeat_final_vowel(word, lo=1, hi=3):
    """ดีีีี-style: repeat the last above-base vowel mark."""
    ABOVE_VOWELS = "\u0e34\u0e35\u0e36\u0e37\u0e31"  # ิ ี ึ ื ั
    for i in range(len(word) - 1, -1, -1):
        if word[i] in ABOVE_VOWELS:
            return word[:i+1] + word[i] * random.randint(lo, hi) + word[i+1:]
    return word

def corrupt_thai(word):
    """
    Apply one SNS-style corruption to a single Thai token.
    Only called on tokens that genuinely differ (raw != norm) so the model
    learns the full range of slang surface forms.
    """
    if not word:
        return word

    #swap: replace a standard particle with an informal variant.
    if word in THAI_PARTICLE_VARIANTS:
        return random.choice(THAI_PARTICLE_VARIANTS[word])

    # Laughter: map ฮาๆ / หัวเราะ -> 555 variants.
    if word in ("ฮาๆ", "ฮา", "หัวเราะ"):
        return random.choice(THAI_LAUGH_TOKENS)

    roll = random.random()
    if roll < 0.30:
        return _repeat_final_char(word)    
    elif roll < 0.50:
        return _repeat_final_vowel(word)   
    elif roll < 0.65 and len(word) > 1:
        return word + word[-1]             # mild doubling
    return word


def augment_thai(data, ratio=THAI_RATIO, factor=THAI_CORRUPT_FACTOR):
    """
    Two-pronged Thai augmentation:

    1. Whole-sentence identity injection (targets FP=404).
       Sentences where every token is already standard (raw == norm for all
       tokens) are re-injected as identity pairs.  This explicitly trains the
       model to output tokens unchanged when the input is already correct,
       which directly suppresses false positives.

    2. Rule-based SNS corruptions (mirrors other languages).
       For sentences that do contain normalisable tokens, we synthesise new
       noisy raw variants so the model sees a wider range of slang forms.
       This prevents the model from treating standard text as slang by giving
       it clearer signal about what real slang looks like.
    """
    n_orig = len(data)

    # --- Part 1: whole-sentence identity injection ---
    # Only use sentences where ALL tokens are already standard so we never
    # introduce label noise (the old per-token extraction loses sentence
    # context, which reduced the signal quality).
    identity_pool = [
        item for item in data
        if all(r == n for r, n in zip(item["raw"], item["norm"]))
    ]
    target_identity = int(n_orig * ratio)
    if len(identity_pool) > target_identity:
        identity_pool = random.sample(identity_pool, target_identity)
    identity_aug = [
        {"raw": list(item["raw"]), "norm": list(item["norm"]),
         "lang": "th", "aug_method": "identity"}
        for item in identity_pool
    ]

    # --- Part 2: rule-based SNS corruption ---
    corrupt_aug = []
    for item in data:
        raw, norm = list(item["raw"]), list(item["norm"])
        for _ in range(factor):
            # Only corrupt tokens that are normalisable (raw != norm at that
            # position); leave already-standard tokens untouched so the
            # model still sees correct surrounding context.
            noisy = [
                corrupt_thai(r) if r != n else r
                for r, n in zip(raw, norm)
            ]
            if noisy != raw:
                corrupt_aug.append({
                    "raw": noisy, "norm": norm,
                    "lang": "th", "aug_method": "rule_corrupt"
                })

    result = data + identity_aug + corrupt_aug
    random.shuffle(result)
    print(
        f"  [th] Original: {n_orig:,} | "
        f"Identity: {len(identity_aug):,} | "
        f"Corrupt: {len(corrupt_aug):,} | "
        f"Total: {len(result):,}"
    )
    return result

# ---------------------------------------------------------------------------
# Japanese augmentation
# ---------------------------------------------------------------------------

JA_ENDINGS = {
    "です": "っす", "ます": "っす", "ている": "てる",
    "ていた": "てた", "している": "してる",
}

def corrupt_japanese(word):
    if not word: return word
    roll = random.random()
    if roll < 0.15: return word + ("w" * random.randint(1, 4))
    elif roll < 0.25: return word + ("～" * random.randint(1, 3))
    elif roll < 0.33 and len(word) > 1: return word + word[-1]
    elif roll < 0.53:
        for f, i in JA_ENDINGS.items():
            if word.endswith(f): return word[:-len(f)] + i
    elif roll < 0.61 and len(word) > 2:
        pos = len(word) - 1
        return word[:pos] + "っ" + word[pos:]
    return word

def augment_japanese(data, factor=DEFAULT_FACTOR):
    result = list(data)
    for item in data:
        norm = list(item["norm"])
        for _ in range(factor):
            noisy = [corrupt_japanese(w) for w in norm]
            if noisy != norm:
                result.append({"raw": noisy, "norm": norm, "lang": "ja", "aug_method": "rule_corrupt"})
    random.shuffle(result)
    print(f"  [ja] Original: {len(data):,} | Added: {len(result)-len(data):,} | Total: {len(result):,}")
    return result

# ---------------------------------------------------------------------------
# Korean augmentation
# ---------------------------------------------------------------------------

KO_FILLERS = ["ㅋ", "ㅎ", "ㅠ", "ㅜ", "~", "..."]

def corrupt_korean(word):
    if not word: return word
    roll = random.random()
    if roll < 0.25: return word + (random.choice(KO_FILLERS) * random.randint(1, 3))
    elif roll < 0.35 and "ㅇ" in word: return word.replace("ㅇ", "0", 1)
    elif roll < 0.43 and len(word) > 1: return word + word[-1]
    elif roll < 0.48 and len(word) <= 3: return word + word
    elif roll < 0.53: return word + "..."
    return word

def augment_korean(data, factor=DEFAULT_FACTOR):
    result = list(data)
    for item in data:
        raw, norm = list(item["raw"]), list(item["norm"])
        for _ in range(factor):
            noisy = [corrupt_korean(r) if r != n else r for r, n in zip(raw, norm)]
            if noisy != raw:
                result.append({"raw": noisy, "norm": norm, "lang": "ko", "aug_method": "rule_corrupt"})
    random.shuffle(result)
    print(f"  [ko] Original: {len(data):,} | Added: {len(result)-len(data):,} | Total: {len(result):,}")
    return result

# ---------------------------------------------------------------------------
# English augmentation
# ---------------------------------------------------------------------------

EN_VOWELS = "aeiou"

def corrupt_english(word):
    if not word: return word
    roll = random.random()
    if roll < 0.15 and word.endswith("ing") and len(word) > 4: return word[:-1] + "'"
    elif roll < 0.25:
        for i, ch in enumerate(word):
            if ch in EN_VOWELS: return word[:i] + (ch * random.randint(2, 4)) + word[i+1:]
    elif roll < 0.33 and word.endswith("e") and len(word) > 3: return word[:-1]
    elif roll < 0.40 and word[0].isupper(): return word.lower()
    elif roll < 0.46 and len(word) > 2:
        i = random.randint(0, len(word)-2)
        w = list(word); w[i], w[i+1] = w[i+1], w[i]; return "".join(w)
    return word

def augment_english(data, factor=DEFAULT_FACTOR):
    result = list(data)
    for item in data:
        norm = list(item["norm"])
        for _ in range(factor):
            noisy = [corrupt_english(w) for w in norm]
            if noisy != norm:
                result.append({"raw": noisy, "norm": norm, "lang": "en", "aug_method": "rule_corrupt"})
    random.shuffle(result)
    print(f"  [en] Original: {len(data):,} | Added: {len(result)-len(data):,} | Total: {len(result):,}")
    return result

# ---------------------------------------------------------------------------
# German augmentation
# ---------------------------------------------------------------------------

DE_UMLAUT = {"ä":"ae","ö":"oe","ü":"ue","Ä":"Ae","Ö":"Oe","Ü":"Ue","ß":"ss"}

def corrupt_german(word):
    if not word: return word
    roll = random.random()
    if roll < 0.25:
        r = word
        for u, rep in DE_UMLAUT.items(): r = r.replace(u, rep)
        if r != word: return r
    elif roll < 0.40 and word[0].isupper(): return word.lower()
    elif roll < 0.50 and word.endswith("e") and len(word) > 3: return word[:-1]
    elif roll < 0.57 and len(word) > 2:
        i = random.randint(0, len(word)-2)
        w = list(word); w[i], w[i+1] = w[i+1], w[i]; return "".join(w)
    return word

def augment_german(data, factor=DEFAULT_FACTOR):
    result = list(data)
    for item in data:
        norm = list(item["norm"])
        for _ in range(factor):
            noisy = [corrupt_german(w) for w in norm]
            if noisy != norm:
                result.append({"raw": noisy, "norm": norm, "lang": "de", "aug_method": "rule_corrupt"})
    random.shuffle(result)
    print(f"  [de] Original: {len(data):,} | Added: {len(result)-len(data):,} | Total: {len(result):,}")
    return result

# ---------------------------------------------------------------------------
# Registry, verification, CLI
# ---------------------------------------------------------------------------

AUGMENTERS = {
    "th": augment_thai,
    "ja": augment_japanese,
    "ko": augment_korean,
    "en": augment_english,
    "de": augment_german,
}

def verify(filepath, n=2):
    print(f"\n[Verification] Sample from {filepath}:")
    by_lang = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            by_lang.setdefault(item.get("lang", "??"), []).append(item)
    for lang, items in sorted(by_lang.items()):
        print(f"  [{lang}] {len(items):,} total")
        for item in items[:n]:
            tag = "IDENTITY" if item["raw"] == item["norm"] else item.get("aug_method", "?").upper()
            print(f"    [{tag}] raw:{item['raw'][:4]}  norm:{item['norm'][:4]}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--langs",  nargs="+", default=DEFAULT_LANGS)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--factor", type=int,   default=DEFAULT_FACTOR)
    p.add_argument("--ratio",  type=float, default=THAI_RATIO)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    random.seed(SEED)
    print("=" * 55)
    print("Input-Variation Augmentation — MultiLexNorm2026")
    print("=" * 55)
    print(f"Languages: {args.langs}  |  Output: {args.output}")
    print()
    all_aug = []
    for lang in args.langs:
        if lang not in AUGMENTERS:
            print(f"  [SKIP] No augmenter for '{lang}'"); continue
        data = load_lang_data(lang)
        if not data: continue
        fn  = AUGMENTERS[lang]
        aug = fn(data, ratio=args.ratio) if lang == "th" else fn(data, factor=args.factor)
        all_aug.extend(aug)
    if all_aug:
        print(); save_jsonl(all_aug, args.output)
        verify(args.output)
        print(f"\nDone. Total: {len(all_aug):,} examples")
    else:
        print("No data generated — check that .jsonl files exist in data/")
