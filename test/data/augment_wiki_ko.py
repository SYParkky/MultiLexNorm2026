"""
augment_wiki_ko.py — Korean-only Wikipedia Corruption Augmentation
Standalone script. Output format matches finetune_byt5-base.py:
    {'raw': 'noisy word', 'norm': 'clean word', 'lang': 'ko', 'corruption': '<rule>'}

Addresses the five FN error types identified in MultiLexNorm KO analysis:
    1. Unregistered slang / abbreviations  (미등록 속어/축약어)
    2. Choseong-only forms                 (초성체, e.g. ㅈㄴ, ㄹㅇ)
    3. Dialect endings                     (방언 어미, e.g. -노, 커엽-)
    4. Trailing jamo noise                 (어말 자모, e.g. 뎈, 앜)
    5. Meaning-substitution slang          (의미 치환, reverse dict)

Usage:
    python augment_wiki_ko.py --samples 5000 --output data/wiki_aug_ko.jsonl

Requirements:
    pip install datasets konlpy
"""

import random
import re
import json
import argparse
import unicodedata
from collections import Counter
from datasets import load_dataset

random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# Hangul utilities
# ══════════════════════════════════════════════════════════════════════════════

CHO_LIST  = list('ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ')
JUNG_LIST = list('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')
JONG_LIST = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ',
             'ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

def decompose_hangul(char):
    code = ord(char) - 0xAC00
    if not (0 <= code <= 11171):
        return None
    cho  = code // (21 * 28)
    jung = (code % (21 * 28)) // 28
    jong = code % 28
    return CHO_LIST[cho], JUNG_LIST[jung], JONG_LIST[jong]

def compose_hangul(cho, jung, jong=''):
    try:
        ci = CHO_LIST.index(cho)
        vi = JUNG_LIST.index(jung)
        ji = JONG_LIST.index(jong)
        return chr(0xAC00 + ci * 21 * 28 + vi * 28 + ji)
    except ValueError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Slang / abbreviation reverse dictionaries
# Direction: clean (wiki) word → noisy (internet) form
# ══════════════════════════════════════════════════════════════════════════════

CLEAN_TO_NOISY_WORD = {
    # Social abbreviations
    '여자친구':         ['여친'],
    '남자친구':         ['남친'],
    '아웃사이더':       ['아싸'],
    '인사이더':         ['인싸'],
    '솔직히':           ['솔까'],
    '그냥':             ['걍'],
    '너무':             ['넘'],
    '매우':             ['존나', 'ㅈㄴ', '겁나'],
    '진짜':             ['ㄹㅇ', '찐짜'],
    '모르겠다':         ['몰루'],
    # Internet/gaming slang
    '불운':             ['언럭키'],
    '행운':             ['럭키'],
    '최고의 선수':      ['GOAT'],
    '애니메이션':       ['애니'],
    '프로필사진':       ['프사'],
    '애니메이션프로필사진': ['애니프사'],
    # Normalized profanity (matching gold annotation style)
    '바보':             ['병신', '빡대갈'],
    '이런':             ['씨발', '시발', 'ㅅㅂ'],
    # Slang verbs / expressions
    '아부하는':         ['빨아대는'],
    '이김':             ['압살함'],
    '망했다':           ['씹창', '개똥땅'],
    '자기들':           ['지들'],
    '먹어라':           ['쳐먹어라'],
    '마시고':           ['쳐먹고'],
}

# Suffix-level: clean ending → noisy ending
# Applied when no exact word match is found
CLEAN_TO_NOISY_SUFFIX = {
    '네':  ['노', '눼'],    # 귀엽네 → 귀엽노 (경상도)
    '다':  ['노'],          # 좋다 → 좋노
    '냐':  ['노'],
}

# Known choseong forms
CLEAN_TO_CHOSEONG = {
    '이런':  'ㅅㅂ',
    '매우':  'ㅈㄴ',
    '진짜':  'ㄹㅇ',
    '인정':  'ㅇㅈ',
    '안녕':  'ㅎㅇ',
    '감사':  'ㄳ',
    '축하':  'ㅊㅋ',
    '바보':  'ㅂㅅ',
    '오케이': 'ㅇㅋ',
}


# ══════════════════════════════════════════════════════════════════════════════
# Corruption rules
# ══════════════════════════════════════════════════════════════════════════════

# ── Rule 1: Slang reverse mapping ─────────────────────────────────────────────

def ko_slang_reverse(word):
    if word in CLEAN_TO_NOISY_WORD:
        return random.choice(CLEAN_TO_NOISY_WORD[word])
    if word in CLEAN_TO_CHOSEONG:
        return CLEAN_TO_CHOSEONG[word]
    for suffix_len in [2, 1]:
        if len(word) >= suffix_len:
            suffix = word[-suffix_len:]
            if suffix in CLEAN_TO_NOISY_SUFFIX:
                return word[:-suffix_len] + random.choice(CLEAN_TO_NOISY_SUFFIX[suffix])
    return word  # unchanged → filtered out by caller


# ── Rule 2: Dialect vowel substitution ────────────────────────────────────────

DIALECT_VOWEL_MAP = {
    'ㅟ': 'ㅓ',   # 귀엽 → 커엽
    'ㅚ': 'ㅔ',   # 외 → 에
}

def ko_dialect_vowel(word):
    candidates = [i for i, ch in enumerate(word)
                  if (d := decompose_hangul(ch)) and d[1] in DIALECT_VOWEL_MAP]
    if not candidates:
        return word
    idx = random.choice(candidates)
    d = decompose_hangul(word[idx])
    new_char = compose_hangul(d[0], DIALECT_VOWEL_MAP[d[1]], d[2])
    if new_char is None:
        return word
    return word[:idx] + new_char + word[idx+1:]


# ── Rule 3: Dialect sentence-final ending ─────────────────────────────────────

def ko_dialect_ending(word):
    if word.endswith('네') and len(word) > 1:
        return word[:-1] + '노'
    if word.endswith('다') and len(word) > 1:
        return word[:-1] + '노'
    if word.endswith('냐') and len(word) > 1:
        return word[:-1] + '노'
    return word


# ── Rule 4: Trailing jamo noise ───────────────────────────────────────────────

TRAILING_JAMO_CHOICES = ['ㅋ', 'ㄱ', 'ㄴ', 'ㄷ', 'ㅎ']

def ko_trail_jamo(word):
    """
    Append a trailing jamo to the last syllable.
    If it has no 종성, compose it in. Otherwise append as a bare jamo.
    e.g. 데 → 뎈, 아 → 앜
    """
    last = word[-1]
    d = decompose_hangul(last)
    trail = random.choice(TRAILING_JAMO_CHOICES)
    if d is None:
        return word + trail
    cho, jung, jong = d
    if jong == '':
        new_char = compose_hangul(cho, jung, trail)
        if new_char:
            return word[:-1] + new_char
    return word + trail


# ── Rule 5: Filler append ─────────────────────────────────────────────────────

KO_FILLERS = ['ㅋ', 'ㅠ', 'ㅜ', 'ㅎ', '~', 'ㅋㅋ', 'ㅠㅠ', '...']

def ko_filler(word):
    return word + random.choice(KO_FILLERS)


# ── Rule 6: Syllable repetition ───────────────────────────────────────────────

def ko_repeat(word):
    if random.random() < 0.3:
        return word * 2
    return word + word[-1]


# ══════════════════════════════════════════════════════════════════════════════
# Rule table
# ══════════════════════════════════════════════════════════════════════════════

KO_RULES = [
    ko_slang_reverse,
    ko_dialect_vowel,
    ko_dialect_ending,
    ko_trail_jamo,
    ko_filler,
    ko_repeat,
]
KO_WEIGHTS = [0.30, 0.15, 0.20, 0.20, 0.10, 0.05]
KO_NAMES   = ['slang_reverse', 'dialect_vowel', 'dialect_ending',
               'trail_jamo', 'filler', 'repeat']

def corrupt_ko(word):
    fn, name = random.choices(list(zip(KO_RULES, KO_NAMES)), weights=KO_WEIGHTS, k=1)[0]
    return fn(word), name


# ══════════════════════════════════════════════════════════════════════════════
# Word filter (same as augment_wiki.py)
# ══════════════════════════════════════════════════════════════════════════════

def is_valid(word):
    if len(word) < 2:
        return False
    if word.isdigit():
        return False
    if all(unicodedata.category(c).startswith('P') for c in word):
        return False
    return True

def tokenize_ko(text):
    try:
        from konlpy.tag import Okt
        return Okt().morphs(text)
    except Exception:
        print("[ko] KoNLPy unavailable — falling back to whitespace tokenization")
        return text.split()


# ══════════════════════════════════════════════════════════════════════════════
# Main generation loop
# ══════════════════════════════════════════════════════════════════════════════

def generate_ko(n_samples, n_variants=3):
    print("[ko] Loading Wikipedia (20231101.ko)...")
    dataset = load_dataset('wikimedia/wikipedia', '20231101.ko',
                           split='train', streaming=True)
    pairs = []
    seen  = set()

    for article in dataset:
        if len(pairs) >= n_samples:
            break
        tokens = tokenize_ko(article.get('text', ''))
        for token in tokens:
            if len(pairs) >= n_samples:
                break
            word = token.strip('.,!?()[]{}「」『』。、…―\'"')
            if not is_valid(word) or word in seen:
                continue
            seen.add(word)
            for _ in range(n_variants):
                corrupted, rule = corrupt_ko(word)
                if corrupted != word:
                    pairs.append({
                        'raw':        corrupted,
                        'norm':       word,
                        'lang':       'ko',
                        'corruption': rule,
                    })

    random.shuffle(pairs)
    return pairs[:n_samples]


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Korean-only Wikipedia augmentation')
    parser.add_argument('--samples',  type=int, default=5000)
    parser.add_argument('--variants', type=int, default=3)
    parser.add_argument('--output',   type=str, default='data/wiki_aug_ko.jsonl')
    args = parser.parse_args()

    pairs = generate_ko(args.samples, args.variants)

    with open(args.output, 'w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"\n✅ {len(pairs)} pairs → {args.output}")

    counts = Counter(p['corruption'] for p in pairs)
    print("\nRule coverage:")
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {name:20s}  {count:5d}  ({count / len(pairs) * 100:.1f}%)")

    print("\nSample output:")
    for p in random.sample(pairs, min(8, len(pairs))):
        print(f"  ({p['corruption']:18s})  {p['norm']!r:20s}  →  {p['raw']!r}")

if __name__ == '__main__':
    main()
