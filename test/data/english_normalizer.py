"""
English YouTube Comment Normalizer
Target: MultiLexNorm English track

Goals:
  1. Normalize common internet slang
  2. Fix lightweight typos
  3. Remove emojis / hashtags / timestamps / mentions
  4. Keep meaning intact
  5. Avoid over-normalization
"""

import re
import json
import random
from typing import Optional

random.seed(42)

# ═══════════════════════════════════════════════════
# 1. SLANG / ABBREVIATION DICTIONARY
# ═══════════════════════════════════════════════════

SLANG_DICT = {

    # reactions
    "fr": "for real",
    "ong": "I swear",
    "idk": "I don't know",
    "imo": "in my opinion",
    "imho": "in my opinion",
    "tbh": "to be honest",
    "ngl": "not gonna lie",
    "irl": "in real life",

    # agreement
    "fax": "facts",
    "facts": "facts",
    "yup": "yes",
    "nah": "no",
    "nope": "no",

    # laughing
    "lmao": "laughing",
    "lmfao": "laughing",
    "rofl": "laughing",
    "lol": "laughing",
    "wtf": "what",
    "wth": "what",

    # people
    "u": "you",
    "ur": "your",
    "tho": "though",

    # youtube/internet
    "yt": "youtube",
    "vid": "video",
    "sub": "subscribe",
    "pls": "please",
    "plz": "please",

    # expressions
    "omg": "oh my god",
    "goated": "great",
    "fire": "great",
    "mid": "average",

    # common misspellings
    "definately": "definitely",
    "definetly": "definitely",
    "seperate": "separate",
    "becuz": "because",
    "cuz": "because",
    "coz": "because",
    "thoo": "though",

    # contractions / casual
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "have to",
    "kinda": "kind of",
    "sorta": "sort of",
}

_SORTED_KEYS = sorted(SLANG_DICT.keys(), key=len, reverse=True)

# ═══════════════════════════════════════════════════
# 2. YOUTUBE CLEANUP
# ═══════════════════════════════════════════════════

YOUTUBE_PATTERNS = [

    # timestamps
    (re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\b'), ''),

    # mentions
    (re.compile(r'@\w+'), ''),

    # hashtags
    (re.compile(r'#\w+'), ''),

    # urls
    (re.compile(r'https?://\S+'), ''),
    (re.compile(r'www\.\S+'), ''),

    # emojis
    (re.compile(
        r'[\U0001F300-\U0001FAFF'
        r'\U00002600-\U000027BF]+'
    ), ''),

]

# ═══════════════════════════════════════════════════
# 3. TYPO FIXES
# ═══════════════════════════════════════════════════

TYPO_PATTERNS = [

    # repeated punctuation
    (re.compile(r'[!]{2,}'), '!'),
    (re.compile(r'[?]{2,}'), '?'),

    # stretched words
    (re.compile(r'(.)\1{3,}'), r'\1\1'),

    # spacing
    (re.compile(r'\s+'), ' '),

]

# ═══════════════════════════════════════════════════
# 4. VALIDATION
# ═══════════════════════════════════════════════════

def is_valid_comment(text: str) -> bool:

    if not isinstance(text, str):
        return False

    text = text.strip()

    if len(text) < 4 or len(text) > 160:
        return False

    if not re.search(r'[a-zA-Z]', text):
        return False

    return True

# ═══════════════════════════════════════════════════
# 5. CLEAN TEXT
# ═══════════════════════════════════════════════════

def clean_text(text: str) -> str:

    text = text.strip()

    for pattern, replacement in YOUTUBE_PATTERNS:
        text = pattern.sub(replacement, text)

    for pattern, replacement in TYPO_PATTERNS:
        text = pattern.sub(replacement, text)

    return text.strip()

# ═══════════════════════════════════════════════════
# 6. TOKEN NORMALIZATION
# ═══════════════════════════════════════════════════

def normalize_token(token: str) -> str:

    lower = token.lower()

    # exact match
    if lower in SLANG_DICT:
        return SLANG_DICT[lower]

    # don't overnormalize short words
    if len(token) <= 2:
        return token

    return token

# ═══════════════════════════════════════════════════
# 7. MAIN NORMALIZATION
# ═══════════════════════════════════════════════════

def normalize_comment(text: str) -> str:

    norm = clean_text(text)

    tokens = norm.split()

    normalized = []

    for token in tokens:
        normalized.append(normalize_token(token))

    norm = " ".join(normalized)

    # final cleanup
    norm = re.sub(r'\s+', ' ', norm).strip()

    return norm

# ═══════════════════════════════════════════════════
# 8. DATASET BUILDER
# ═══════════════════════════════════════════════════

def build_dataset(comments, max_samples=20000):

    dataset = []
    seen = set()

    for raw in comments:

        raw = clean_text(raw)

        if not is_valid_comment(raw):
            continue

        norm = normalize_comment(raw)

        # skip unchanged
        if raw == norm:
            continue

        # avoid extreme shortening
        if len(norm) < len(raw) * 0.4:
            continue

        pair = (raw, norm)

        if pair in seen:
            continue

        seen.add(pair)

        dataset.append({
            "raw": raw,
            "norm": norm,
            "lang": "en"
        })

        if len(dataset) >= max_samples:
            break

    return dataset

# ═══════════════════════════════════════════════════
# 9. TESTS
# ═══════════════════════════════════════════════════

TEST_CASES = [

    ("bro this vid is fireeee 🔥🔥", 
     "friend this video is great"),

    ("idk why ppl hate this lol", 
     "I don't know why ppl hate this laughing"),

    ("omg this is goated fr", 
     "oh my god this is great for real"),

    ("wtf did i just watch 😂", 
     "what did i just watch"),

    ("plz upload more vids!!", 
     "please upload more videos!"),

    ("nah bro ur wrong", 
     "no friend your wrong"),

]

def run_tests():

    print("\n" + "="*60)
    print("SELF TEST")
    print("="*60)

    for raw, expected in TEST_CASES:

        result = normalize_comment(raw)

        print(f"RAW : {raw}")
        print(f"NORM: {result}")
        print(f"EXP : {expected}")
        print()

# ═══════════════════════════════════════════════════
# 10. MAIN
# ═══════════════════════════════════════════════════

if __name__ == "__main__":

    with open("youtube_comments_en.json", "r", encoding="utf-8") as f:
        comments = json.load(f)

    dataset = build_dataset(comments)

    with open("english_normalized.jsonl", "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Built dataset: {len(dataset)}")

    run_tests()
