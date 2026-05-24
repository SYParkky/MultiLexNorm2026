import random
import re
import json
import argparse
from collections import Counter

random.seed(42)

# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def is_valid_text(text):

    if not text:
        return False

    text = text.strip()

    if len(text) < 5:
        return False

    if len(text) > 80:
        return False

    if not re.search(r"[가-힣]", text):
        return False

    return True


def tokenize(text):
    return text.split()


# ═══════════════════════════════════════════════════════════════
# Realistic Korean Internet Noise
# ═══════════════════════════════════════════════════════════════

CLEAN_TO_NOISY = {

    '진짜': ['ㄹㅇ', '레알', '진짜루'],
    '너무': ['넘', '개', '겁나'],
    '정말': ['진짜', 'ㄹㅇ'],

    '좋아': ['조아', '좋앙'],
    '좋다': ['좋음', '굳'],

    '웃기다': ['개웃기다', '존웃'],
    '웃겨': ['개웃겨', '존웃'],

    '싫어': ['실어', '극혐'],

    '인정': ['ㅇㅈ'],
    '맞아': ['ㅇㅇ', 'ㄹㅇ'],
    '아니야': ['ㄴㄴ'],

    '몰라': ['몰루', 'ㅁㄹ'],

    '고마워': ['ㄱㅅ'],
    '미안해': ['ㅁㅇ'],
    '축하해': ['ㅊㅋ'],

    '귀엽다': ['커엽다'],
    '귀여워': ['커여워'],

    '없어': ['업서'],
    '있어': ['이써'],

    '어떻게': ['어케', '어캐'],
    '이렇게': ['이케'],

    '그냥': ['걍'],
}

FILLERS = [
    'ㅋㅋ',
    'ㅋㅋㅋ',
    'ㅠㅠ',
    'ㄹㅇ',
    '...',
]

DOE_DAE_RULES = [
    ('돼', '되'),
    ('됐', '됬'),
    ('좋아', '조아'),
    ('싫어', '실어'),
    ('많이', '마니'),
]

SPACING_RULES = [
    ('할 수 있어', '할수있어'),
    ('볼 수 있어', '볼수있어'),
    ('생각해 보면', '생각해보면'),
    ('진짜 웃기다', '진짜웃기다'),
    ('너무 좋아', '너무좋아'),
    ('안 돼', '안돼'),
]

RULE_NAMES = [
    'slang',
    'doe_dae',
    'spacing',
    'filler',
]

RULE_WEIGHTS = [
    0.45,
    0.20,
    0.15,
    0.20,
]


# ═══════════════════════════════════════════════════════════════
# Corruption Functions
# ═══════════════════════════════════════════════════════════════

def apply_slang(token):

    if token in CLEAN_TO_NOISY:
        return random.choice(CLEAN_TO_NOISY[token])

    return token


def apply_doe_dae(text):

    for clean, noisy in DOE_DAE_RULES:

        if clean in text and random.random() < 0.5:
            text = text.replace(clean, noisy, 1)

    return text


def apply_spacing(text):

    for clean, noisy in SPACING_RULES:

        if clean in text and random.random() < 0.5:
            text = text.replace(clean, noisy, 1)

    return text


# ═══════════════════════════════════════════════════════════════
# Sentence-Level Corruption
# ═══════════════════════════════════════════════════════════════

def corrupt_sentence(text):

    tokens = tokenize(text)

    applied = []

    if not tokens:
        return text, applied

    n_changes = random.randint(1, min(3, len(tokens)))

    indices = random.sample(
        range(len(tokens)),
        n_changes
    )

    for idx in indices:

        rule = random.choices(
            RULE_NAMES,
            weights=RULE_WEIGHTS,
            k=1
        )[0]

        original = tokens[idx]

        if rule == 'slang':
            tokens[idx] = apply_slang(tokens[idx])

        if tokens[idx] != original:
            applied.append(rule)

    sentence = ' '.join(tokens)

    before = sentence

    sentence = apply_doe_dae(sentence)

    if sentence != before:
        applied.append('doe_dae')

    before = sentence

    sentence = apply_spacing(sentence)

    if sentence != before:
        applied.append('spacing')

    if random.random() < 0.35:

        sentence += random.choice(FILLERS)

        applied.append('filler')

    return sentence, list(set(applied))


# ═══════════════════════════════════════════════════════════════
# Load REAL YouTube Comments
# ═══════════════════════════════════════════════════════════════

def load_youtube_comments(path):

    with open(path, "r", encoding="utf-8") as f:
        comments = json.load(f)

    cleaned = []

    seen = set()

    for text in comments:

        if not isinstance(text, str):
            continue

        text = text.strip()

        # URL 제거
        text = re.sub(r"http\S+", "", text)

        # 공백 정리
        text = re.sub(r"\s+", " ", text)

        if not is_valid_text(text):
            continue

        # 과도한 ㅋㅋ 제거
        if text.count("ㅋ") > 15:
            continue

        # 중복 제거
        if text in seen:
            continue

        seen.add(text)

        cleaned.append(text)

    print(f"\n[ko] loaded {len(cleaned)} real comments")

    print("\nSample comments:\n")

    for x in random.sample(cleaned, min(10, len(cleaned))):
        print(x)

    return cleaned


# ═══════════════════════════════════════════════════════════════
# Generate Dataset
# ═══════════════════════════════════════════════════════════════

def generate_dataset(comments, n_samples=10000):

    dataset = []

    seen_pairs = set()

    for text in comments:

        if len(dataset) >= n_samples:
            break

        noisy, rules = corrupt_sentence(text)

        if noisy == text:
            continue

        pair_key = (noisy, text)

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)

        dataset.append({
            "raw": noisy,
            "norm": text,
            "lang": "ko",
            "corruption": rules
        })

    return dataset


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        type=str,
        default='youtube_comments.json'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='augment_ko_real.jsonl'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=5000
    )

    args = parser.parse_args()

    comments = load_youtube_comments(args.input)

    dataset = generate_dataset(
        comments=comments,
        n_samples=args.samples
    )

    with open(args.output, 'w', encoding='utf-8') as f:

        for row in dataset:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + '\n'
            )

    print(f"\n✅ saved {len(dataset)} samples")

    print(f"→ {args.output}")

    counts = Counter()

    for row in dataset:

        for r in row['corruption']:
            counts[r] += 1

    total = len(dataset)

    print("\nRule coverage:\n")

    for k, v in sorted(
        counts.items(),
        key=lambda x: -x[1]
    ):

        print(
            f"{k:15s} "
            f"{v:5d} "
            f"({v/total*100:.1f}%)"
        )

    print("\nSample outputs:\n")

    for x in random.sample(
        dataset,
        min(10, len(dataset))
    ):

        print("RAW :", x['raw'])
        print("NORM:", x['norm'])
        print("RULE:", x['corruption'])
        print()


if __name__ == '__main__':
    main()