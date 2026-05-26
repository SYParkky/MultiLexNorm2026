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
    "tbh": "to be honest",
    "ngl": "not gonna lie",
    "irl": "in real life",
    "smh": "shaking my head",

    # agreement / disagreement
    "yup": "yes",
    "yep": "yes",
    "nah": "no",
    "nope": "no",
    "fs": "for sure",
    "ofc": "of course",
    "obv": "obviously",
    "obvs": "obviously",

    # laughing / reactions
    "lmao": "laughing my ass off",
    "lmfao": "laughing my fucking ass off",
    "rofl": "rolling on the floor laughing",
    "lol": "laughing out loud",
    "lmaoo": "laughing my ass off",
    "lmaooo": "laughing my ass off",

    # wtf / wth 
    "wtf": "what the fuck",
    "wth": "what the hell",

    # people / pronouns
    "u": "you",
    "ur": "your",
    "ya": "you",
    "ure": "you are",
    "wut": "what",
    "wot": "what",


    # youtube / internet
    "yt": "youtube",
    "vid": "video",
    "vids": "videos",
    "sub": "subscribe",
    "subs": "subscribers",
    "pls": "please",
    "plz": "please",
    "rn": "right now",
    "pov": "point of view",
    "lmk": "let me know",
    "dm": "direct message",
    "fyi": "for your information",

    # expressions
    "omg": "oh my god",
    "omfg": "oh my god",
    "goated": "greatest of all time",
    "goat": "greatest of all time",
    "fire": "amazing",
    "mid": "average",
    "sus": "suspicious",
    "sheesh": "wow",
    "periodt": "period",
    "iykyk": "if you know you know",
  
    # common misspellings / phonetic
    "definately": "definitely",
    "definetly": "definitely",
    "seperate": "separate",
    "becuz": "because",
    "cuz": "because",
    "coz": "because",
    "bcuz": "because",
    "bc": "because",
    "prolly": "probably",
    "probs": "probably",
    "prob": "probably",
    "rly": "really",
    "rlly": "really",
    "realy": "really",
    "ppl": "people",
    "idc": "I don't care",
    "ik": "I know",
    "ikr": "I know right",
    "dw": "don't worry",
    "nvm": "never mind",
    "thx": "thanks",
    "ty": "thank you",
    "tysm": "thank you so much",
    "ily": "I love you",
    "b4": "before",
    "gr8": "great",
    "l8r": "later",
    "asap": "as soon as possible",
    "np": "no problem",
    "omw": "on my way",
    "wdym": "what do you mean",
    "wym": "what do you mean",
    "brb": "be right back",
    "ttyl": "talk to you later",

    # contractions / casual speech
    "tryna": "trying to",
    "finna": "about to",
    "shoulda": "should have",
    "coulda": "could have",
    "woulda": "would have",
    "dunno": "don't know",
    "gimme": "give me",
    "lemme": "let me",
    "imma": "I'm going to",
    "ima": "I'm going to",
    "tho": "though",
}

_SORTED_KEYS = sorted(SLANG_DICT.keys(), key=len, reverse=True)

# ═══════════════════════════════════════════════════
# 2. MULTI-WORD PHRASE PATTERNS
# ═══════════════════════════════════════════════════

PHRASE_PATTERNS = [
    (re.compile(r'\bno\s+cap\b', re.I), 'for real'),
    (re.compile(r'\bon\s+god\b', re.I), 'I swear'),
    (re.compile(r'\bfr\s+fr\b', re.I), 'for real'),
    (re.compile(r'\bnot\s+gonna\s+lie\b', re.I), 'to be honest'),
    (re.compile(r'\bfor\s+real\s+for\s+real\b', re.I), 'for real'),
    (re.compile(r'\btouch\s+grass\b', re.I), 'go outside'),
    (re.compile(r'\bpeak\s+(?:fiction|cinema|content)\b', re.I), 'excellent content'),
    (re.compile(r"\bain't\s+no\s+way\b", re.I), "I can't believe"),
    (re.compile(r'\blet\s+him\s+cook\b', re.I), 'let him continue'),
    (re.compile(r'\bin\s+my\s+\w+\s+era\b', re.I), 'currently focused on this'),
    (re.compile(r'\bwhy\s+is\s+nobody\s+talking\s+about\b', re.I), 'people should discuss'),
    (re.compile(r'\bsend\s+help\b', re.I), 'I need help'),
    (re.compile(r'\bright\s+in\s+the\s+feels\b', re.I), 'emotionally moving'),
    (re.compile(r'\bpov\s*:', re.I), 'point of view:'),
]

# ═══════════════════════════════════════════════════
# 3. YOUTUBE CLEANUP
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
        r'\U00002600-\U000027BF'
        r'\U0001F900-\U0001F9FF'
        r'\U00002300-\U000023FF'
        r'\uFE00-\uFE0F'
        r'\u200d'
        r'\u20E3'
        r']+'
    ), ''),

    # boilerplate spam
    (re.compile(r'\blike\s+if\s+you\s+agree\b', re.I), ''),
    (re.compile(r'\bLIKE\s+AND\s+SUBSCRIBE\b'), ''),
    (re.compile(r'\bnotification\s+squad\b', re.I), ''),
    (re.compile(r'\b(first|second|third)\s*!?\s*$', re.I), ''),
]

# ═══════════════════════════════════════════════════
# 4. TYPO FIXES
# ═══════════════════════════════════════════════════

TYPO_PATTERNS = [

    # repeated punctuation
    (re.compile(r'[!]{2,}'), '!'),
    (re.compile(r'[?]{2,}'), '?'),
    (re.compile(r'[.]{4,}'), '...'),

    # stretched words: keep max 2 of any char
    (re.compile(r'(.)\1{2,}'), r'\1\1'),

    # spacing
    (re.compile(r'\s+'), ' '),
]

# ═══════════════════════════════════════════════════
# 5. VALIDATION
# ═══════════════════════════════════════════════════

def is_valid_comment(text: str) -> bool:

    if not isinstance(text, str):
        return False

    text = text.strip()

    if len(text) < 4 or len(text) > 160:
        return False

    if not re.search(r'[a-zA-Z]', text):
        return False

    # reject mostly non-ASCII (foreign language)
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if ascii_chars / len(text) < 0.7:
        return False

    return True

# ═══════════════════════════════════════════════════
# 6. CLEAN TEXT
# ═══════════════════════════════════════════════════

def clean_text(text: str) -> str:

    text = text.strip()

    # phrase-level first
    for pattern, replacement in PHRASE_PATTERNS:
        text = pattern.sub(replacement, text)

    for pattern, replacement in YOUTUBE_PATTERNS:
        text = pattern.sub(replacement, text)

    for pattern, replacement in TYPO_PATTERNS:
        text = pattern.sub(replacement, text)

    return text.strip()

# ═══════════════════════════════════════════════════
# 7. TOKEN NORMALIZATION
# ═══════════════════════════════════════════════════

def normalize_token(token: str) -> str:

    lower = token.lower()

    # exact slang match
    if lower in SLANG_DICT:
        return SLANG_DICT[lower]

    # skip very short tokens
    if len(token) <= 2:
        return token

    return token

# ═══════════════════════════════════════════════════
# 8. MAIN NORMALIZATION
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
# 9. DATASET BUILDER
# ═══════════════════════════════════════════════════

def build_dataset(comments, max_samples=5000):

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
# 10. TESTS
# ═══════════════════════════════════════════════════
# skip

# ═══════════════════════════════════════════════════
# 11. MAIN
# ═══════════════════════════════════════════════════

if __name__ == "__main__":

    with open("youtube_comments.json", "r", encoding="utf-8") as f:
        comments = json.load(f)

    dataset = build_dataset(comments)

    with open("english_normalized.jsonl", "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Built dataset: {len(dataset)}")


