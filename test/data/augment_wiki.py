"""
augment_wiki.py — Wikipedia Corruption Augmentation 
Generates synthetic (noisy_input, clean_target) pairs for lexical normalization.

Output format matches finetune_byt5-base.py:
    {'raw': 'noisy word', 'norm': 'clean word', 'lang': 'ko'}

Usage:
    python data/augment_wiki.py --langs ko ja en de --samples 5000 --output data/wiki_aug.jsonl

Requirements:
    pip install datasets konlpy fugashi unidic-lite
"""

import random
import re
import json
import argparse
import unicodedata

from datasets import load_dataset

random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# Tokenizers
# ══════════════════════════════════════════════════════════════════════════════

def tokenize_ko(text):
    """Korean morpheme tokenization via KoNLPy/Okt. Falls back to whitespace."""
    try:
        from konlpy.tag import Okt
        return Okt().morphs(text)
    except Exception:
        print("[ko] KoNLPy unavailable — falling back to whitespace tokenization")
        return text.split()

def tokenize_ja(text):
    """Japanese tokenization via fugashi. Falls back to whitespace."""
    try:
        import fugashi
        tagger = fugashi.Tagger()
        return [word.surface for word in tagger(text)]
    except Exception:
        print("[ja] fugashi unavailable — falling back to whitespace tokenization")
        return text.split()

def tokenize_default(text):
    return text.split()

TOKENIZERS = {
    'ko': tokenize_ko,
    'ja': tokenize_ja,
    'en': tokenize_default,
    'de': tokenize_default,
}


# ══════════════════════════════════════════════════════════════════════════════
# Shared utility corruptions
# ══════════════════════════════════════════════════════════════════════════════

def corrupt_delete_char(word):
    """Delete a random non-first character."""
    if len(word) > 2:
        i = random.randint(1, len(word) - 1)
        return word[:i] + word[i+1:]
    return word

def corrupt_swap_chars(word):
    """Swap two adjacent characters (classic typo)."""
    if len(word) > 2:
        i = random.randint(0, len(word) - 2)
        w = list(word)
        w[i], w[i+1] = w[i+1], w[i]
        return ''.join(w)
    return word


# ══════════════════════════════════════════════════════════════════════════════
# Language-specific corruption rules + weights
# ══════════════════════════════════════════════════════════════════════════════

# ── Korean ────────────────────────────────────────────────────────────────────

KO_FILLERS = ['ㅋ', 'ㅠ', 'ㅜ', 'ㅎ', '~', '...']

def _decompose_hangul(char):
    """Break a Hangul syllable into its jamo components."""
    code = ord(char) - 0xAC00
    if not (0 <= code <= 11171):
        return None
    CHO  = list('ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ')
    JUNG = list('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')
    JONG = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ',
            'ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
    cho  = code // (21 * 28)
    jung = (code % (21 * 28)) // 28
    jong = code % 28
    return CHO[cho] + JUNG[jung] + JONG[jong]

def ko_filler(w):       return w + random.choice(KO_FILLERS)
def ko_repeat(w):       return w * 2
def ko_decompose(w):
    for i in range(len(w) - 1, -1, -1):
        d = _decompose_hangul(w[i])
        if d:
            return w[:i] + d
    return w
def ko_num_sub(w):
    result = []
    for ch in w:
        d = _decompose_hangul(ch)
        if d and 'ㅇ' in d:
            result.append(d.replace('ㅇ', '0'))
        else:
            result.append(ch)
    return ''.join(result)
def ko_spacing(w):      return corrupt_delete_char(w)   # simulate missing space
def ko_trail_jamo(w):
    """Append the last jamo of the word — e.g. 나비 → 나비ㅣ, 나방 → 나방ㅇ."""
    last = w[-1]
    d = _decompose_hangul(last)
    if d:
        # if there's a 종성, use it; otherwise use the 중성 (vowel)
        trail = d[2] if d[2] else d[1]
        return w + trail
    # non-Hangul: just append the last character as-is
    return w + last

KO_RULES   = [ko_filler, ko_repeat, ko_decompose, ko_num_sub, ko_spacing, ko_trail_jamo]
KO_WEIGHTS = [0.25,      0.05,      0.20,         0.10,       0.20,       0.20]
KO_NAMES   = ['filler',  'repeat',  'decompose',  'num_sub',  'spacing',  'trail_jamo']


# ── Japanese ──────────────────────────────────────────────────────────────────

JA_COLLOQUIAL = {
    'ている': 'てる', 'てしまう': 'ちゃう', 'でしまう': 'じゃう',
    'です': 'っす',   'ます': 'っす',
}

def ja_kana_swap(w):
    result = []
    for ch in w:
        code = ord(ch)
        if 0x3041 <= code <= 0x3096:    # hiragana → katakana
            result.append(chr(code + 0x60))
        elif 0x30A1 <= code <= 0x30F6:  # katakana → hiragana
            result.append(chr(code - 0x60))
        else:
            result.append(ch)
    return ''.join(result)

def ja_long_vowel(w):
    if len(w) >= 2:
        i = random.randint(1, len(w) - 1)
        return w[:i] + 'ー' + w[i:]
    return w

def ja_colloquial(w):
    for formal, casual in JA_COLLOQUIAL.items():
        if formal in w:
            return w.replace(formal, casual, 1)
    return w + 'w'   # fallback: append 笑 abbreviation

def ja_repeat_char(w):  return w + w[-1]
def ja_swap(w):         return corrupt_swap_chars(w)

JA_RULES   = [ja_kana_swap, ja_long_vowel, ja_colloquial, ja_repeat_char, ja_swap]
JA_WEIGHTS = [0.30,         0.20,          0.20,          0.20,           0.10]
JA_NAMES   = ['kana_swap',  'long_vowel',  'colloquial',  'repeat_char',  'swap_chars']


# ── English ───────────────────────────────────────────────────────────────────

VOWELS = set('aeiouAEIOU')
EN_CONTRACTIONS = {
    'want to': 'wanna', 'going to': 'gonna',
    'got to':  'gotta', 'kind of':  'kinda', 'out of': 'outta',
}

def en_repeat_vowel(w):
    indices = [i for i, c in enumerate(w) if c in VOWELS]
    if indices:
        i = random.choice(indices)
        return w[:i] + w[i] * random.randint(2, 4) + w[i+1:]
    return w

def en_lowercase(w):    return w.lower()
def en_drop_g(w):
    if w.lower().endswith('ing'):
        return w[:-1] + "'"
    return w
def en_contraction(w):
    for formal, casual in EN_CONTRACTIONS.items():
        if formal in w.lower():
            return re.sub(re.escape(formal), casual, w, flags=re.IGNORECASE)
    return w + w[-1]
def en_delete(w):       return corrupt_delete_char(w)
def en_swap(w):         return corrupt_swap_chars(w)

EN_RULES   = [en_repeat_vowel, en_lowercase, en_drop_g, en_contraction, en_delete, en_swap]
EN_WEIGHTS = [0.25,            0.20,         0.15,      0.10,           0.20,      0.10]
EN_NAMES   = ['repeat_vowel',  'lowercase',  'drop_g',  'contraction',  'delete',  'swap_chars']


# ── German ────────────────────────────────────────────────────────────────────

DE_UMLAUT = {'ä':'ae','ö':'oe','ü':'ue','Ä':'Ae','Ö':'Oe','Ü':'Ue','ß':'ss'}

def de_umlaut(w):
    for u, r in DE_UMLAUT.items():
        w = w.replace(u, r)
    return w
def de_lowercase(w):    return w.lower()
def de_repeat_vowel(w):
    indices = [i for i, c in enumerate(w) if c.lower() in 'aeiouäöü']
    if indices:
        i = random.choice(indices)
        return w[:i] + w[i] * random.randint(2, 3) + w[i+1:]
    return w
def de_drop_e(w):
    if w.endswith('e') and len(w) > 3:
        return w[:-1]
    return w
def de_swap(w):         return corrupt_swap_chars(w)

DE_RULES   = [de_umlaut, de_lowercase, de_repeat_vowel, de_drop_e, de_swap]
DE_WEIGHTS = [0.30,      0.25,         0.20,            0.15,      0.10]
DE_NAMES   = ['umlaut',  'lowercase',  'repeat_vowel',  'drop_e',  'swap_chars']


# ── Dispatcher ────────────────────────────────────────────────────────────────

LANG_CONFIG = {
    'ko': (KO_RULES, KO_WEIGHTS, KO_NAMES),
    'ja': (JA_RULES, JA_WEIGHTS, JA_NAMES),
    'en': (EN_RULES, EN_WEIGHTS, EN_NAMES),
    'de': (DE_RULES, DE_WEIGHTS, DE_NAMES),
}

WIKI_CONFIG = {
    'ko': ('wikimedia/wikipedia', '20231101.ko'),
    'ja': ('wikimedia/wikipedia', '20231101.ja'),
    'en': ('wikimedia/wikipedia', '20231101.en'),
    'de': ('wikimedia/wikipedia', '20231101.de'),
}

def corrupt(word, lang):
    """Pick one weighted-random rule and apply it. Returns (corrupted, rule_name)."""
    rules, weights, names = LANG_CONFIG[lang]
    fn, name = random.choices(list(zip(rules, names)), weights=weights, k=1)[0]
    return fn(word), name


# ══════════════════════════════════════════════════════════════════════════════
# Word filter
# ══════════════════════════════════════════════════════════════════════════════

def is_valid(word):
    if len(word) < 2:                                                   return False
    if word.isdigit():                                                  return False
    if all(unicodedata.category(c).startswith('P') for c in word):     return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Main generation loop
# ══════════════════════════════════════════════════════════════════════════════

def generate(lang, n_samples, n_variants=3):
    """
    Returns list of dicts — format matches finetune_byt5-base.py exactly:
        'raw'        = noisy input  (what the model sees)
        'norm'       = clean target (what the model outputs) — NEVER changed
        'lang'       = language code
        'corruption' = rule name (for ablation/analysis)
    """
    tokenize        = TOKENIZERS[lang]
    ds_name, ds_cfg = WIKI_CONFIG[lang]

    print(f"[{lang}] Loading Wikipedia ({ds_cfg})...")
    dataset = load_dataset(ds_name, ds_cfg, split='train', streaming=True)

    pairs = []
    seen  = set()

    for article in dataset:
        if len(pairs) >= n_samples:
            break

        tokens = tokenize(article.get('text', ''))

        for token in tokens:
            if len(pairs) >= n_samples:
                break

            word = token.strip('.,!?()[]{}「」『』。、…―\'"')

            if not is_valid(word) or word in seen:
                continue
            seen.add(word)

            for _ in range(n_variants):
                corrupted, rule = corrupt(word, lang)
                if corrupted != word:
                    pairs.append({
                        'raw':        corrupted,
                        'norm':       word,
                        'lang':       lang,
                        'corruption': rule
                    })

    random.shuffle(pairs)
    return pairs[:n_samples]


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Wikipedia corruption augmentation v2')
    parser.add_argument('--langs',    nargs='+', default=['ko', 'ja', 'en', 'de'])
    parser.add_argument('--samples',  type=int,  default=5000, help='Pairs per language')
    parser.add_argument('--variants', type=int,  default=3,    help='Corruption variants per word')
    parser.add_argument('--output',   type=str,  default='data/wiki_aug.jsonl')
    args = parser.parse_args()

    all_pairs = []
    for lang in args.langs:
        if lang not in LANG_CONFIG:
            print(f"[!] Unsupported language: {lang} — skipping")
            continue
        pairs = generate(lang, args.samples, args.variants)
        print(f"[{lang}] {len(pairs)} pairs generated")
        all_pairs.extend(pairs)

    with open(args.output, 'w', encoding='utf-8') as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"\n✅ {len(all_pairs)} total pairs → {args.output}")
    print("\nSample output:")
    for p in all_pairs[:5]:
        print(f"  [{p['lang']}] ({p['corruption']:12s})  {p['raw']!r:20s} → {p['norm']!r}")

if __name__ == '__main__':
    main()

