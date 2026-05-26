"""
English YouTube Comment Normalizer
Target: MultiLexNorm English track

Features:
  1. Slang normalization
  2. Abbreviation expansion
  3. Profanity normalization
  4. Repeated-character normalization
  5. Elongated typo normalization
  6. YouTube cleanup
  7. Safe normalization
"""

import json
import re
import random

random.seed(42)

# ═══════════════════════════════════════════════════════
# 1. SLANG / ABBREVIATION DICTIONARY
# ═══════════════════════════════════════════════════════

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
    "yup": "yes",
    "nah": "no",
    "nope": "no",

    # laughing
    "lmao": "laughing my ass off",
    "lmfao": "laughing my fucking ass off",
    "lol": "laughing out loud",
    "rofl": "rolling on the floor laughing",

    # expressions
    "wtf": "what the fuck",
    "wth": "what the hell",
    "omg": "oh my god",

    # people
    "u": "you",
    "ur": "your",
    "ya": "you",

    # youtube
    "yt": "youtube",
    "vid": "video",
    "vids": "videos",
    "pls": "please",
    "plz": "please",
    "sub": "subscribe",

    # internet slang
    "goated": "great",
    "fire": "great",
    "mid": "average",

    # abbreviations
    "rn": "right now",
    "bc": "because",
    "ppl": "people",
    "smh": "disappointed",
    "idc": "I do not care",
    "ik": "I know",
    "ikr": "I know right",
    "dw": "do not worry",
    "nvm": "never mind",
    "asap": "as soon as possible",
    "thx": "thanks",
    "ty": "thank you",
    "ily": "I love you",

    # casual english
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "have to",
    "kinda": "kind of",
    "sorta": "sort of",
    "tho": "though",

    # misspellings
    "definately": "definitely",
    "definetly": "definitely",
    "seperate": "separate",
    "becuz": "because",
    "cuz": "because",
    "coz": "because",
}

# ═══════════════════════════════════════════════════════
# 2. ELONGATED WORDS
# ═══════════════════════════════════════════════════════

ELONGATED_WORDS = {

    "brooo": "bro",
    "broooo": "bro",

    "nahhh": "nah",
    "nahhhh": "nah",

    "yesss": "yes",
    "yessss": "yes",

    "noooo": "no",
    "nooooo": "no",

    "plsss": "please",
    "plssss": "please",
    "plzz": "please",
    "plzzz": "please",

    "omggg": "omg",
    "omgggg": "omg",

    "lolll": "lol",
    "lmfaooo": "lmfao",
}

# ═══════════════════════════════════════════════════════
# 3. YOUTUBE CLEANUP
# ═══════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════
# 4. PROFANITY NORMALIZATION
# ═══════════════════════════════════════════════════════

PROFANITY_PATTERNS = [

    # fuck
    (re.compile(r'\bf+u*c*k+\b', re.I), 'fuck'),
    (re.compile(r'\bf+k+\b', re.I), 'fuck'),
    (re.compile(r'\bf+u+k+\b', re.I), 'fuck'),
    (re.compile(r'\bf+u+c+c+\b', re.I), 'fuck'),

    # shit
    (re.compile(r'\bs+h+i+t+\b', re.I), 'shit'),
    (re.compile(r'\bsh+t+\b', re.I), 'shit'),

    # bitch
    (re.compile(r'\bb+i+t+c+h+\b', re.I), 'bitch'),
    (re.compile(r'\bb+t+c+h+\b', re.I), 'bitch'),

    # damn
    (re.compile(r'\bd+a+m+n+\b', re.I), 'damn'),
]

# ═══════════════════════════════════════════════════════
# 5. TYPO PATTERNS
# ═══════════════════════════════════════════════════════

TYPO_PATTERNS = [

    # punctuation
    (re.compile(r'[!]{2,}'), '!'),
    (re.compile(r'[?]{2,}'), '?'),

    # whitespace
    (re.compile(r'\s+'), ' '),
]

# ═══════════════════════════════════════════════════════
# 6. VALIDATION
# ═══════════════════════════════════════════════════════

def is_valid_comment(text):

    if not isinstance(text, str):
        return False

    text = text.strip()

    if len(text) < 4 or len(text) > 200:
        return False

    if not re.search(r'[a-zA-Z]', text):
        return False

    return True

# ═══════════════════════════════════════════════════════
# 7. CLEAN TEXT
# ═══════════════════════════════════════════════════════

def clean_text(text):

    text = text.strip()

    for pattern, replacement in YOUTUBE_PATTERNS:
        text = pattern.sub(replacement, text)

    for pattern, replacement in TYPO_PATTERNS:
        text = pattern.sub(replacement, text)

    return text.strip()

# ═══════════════════════════════════════════════════════
# 8. REPEATED CHARACTER NORMALIZATION
# ═══════════════════════════════════════════════════════

def normalize_repeated_chars(word):

    if len(word) <= 3:
        return word

    # fireeeee -> fire
    # broooo -> bro
    word = re.sub(r'(.)\1{2,}', r'\1', word)

    return word

# ═══════════════════════════════════════════════════════
# 9. TOKEN NORMALIZATION
# ═══════════════════════════════════════════════════════

def normalize_token(token):
    lower = token.lower()
    
    if lower in ELONGATED_WORDS:
        lower = ELONGATED_WORDS[lower]
    
    lower = normalize_repeated_chars(lower)  # 순서 변경
    
    if lower in SLANG_DICT:
        return SLANG_DICT[lower]
    
    return lower  # 항상 정규화된 결과 반환

# ═══════════════════════════════════════════════════════
# 10. MAIN NORMALIZATION
# ═══════════════════════════════════════════════════════

def normalize_comment(text):

    norm = clean_text(text)

    # profanity normalization
    for pattern, replacement in PROFANITY_PATTERNS:
        norm = pattern.sub(replacement, norm)

    tokens = norm.split()

    normalized = []

    for token in tokens:
        normalized.append(normalize_token(token))

    norm = " ".join(normalized)

    # final cleanup
    norm = re.sub(r'\s+', ' ', norm).strip()

    return norm

# ═══════════════════════════════════════════════════════
# 11. DATASET BUILDER
# ═══════════════════════════════════════════════════════

def build_dataset(comments, max_samples=5000):

    dataset = []
    seen = set()

    for raw in comments:

        raw = clean_text(raw)

        if not is_valid_comment(raw):
            continue

        norm = normalize_comment(raw)

        # skip unchanged
        if raw.lower() == norm.lower():
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

# ═══════════════════════════════════════════════════════
# 12. MAIN
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":

    with open("youtube_comments_en.json", "r", encoding="utf-8") as f:
        comments = json.load(f)

    print(f"Loaded comments: {len(comments)}")

    dataset = build_dataset(comments, max_samples=5000)

    print(f"Built dataset: {len(dataset)}")

    with open("english_normalized.jsonl", "w", encoding="utf-8") as f:

        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Saved -> english_normalized.jsonl")

