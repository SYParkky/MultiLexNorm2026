import json
import re
import random
import argparse

random.seed(42)

# ═══════════════════════════════════════════════════════════════
# Normalization Rules
# ═══════════════════════════════════════════════════════════════

SLANG_RULES = {

    # internet slang
    'ㄹㅇ': '정말',
    '레알': '정말',

    '개': '매우',
    '존나': '매우',
    '존나게':'매우'
    'ㅈㄴ': '매우'
    '겁나': '매우',

    'ㅇㅈ': '인정',
    'ㄴㄴ': '아니',
    'ㅇㅇ': '응',

    'ㅁㅊ': '미친',
    'ㅅㅂ': '짜증난',

    'ㄱㅅ': '고마워',
    'ㅊㅋ': '축하해',

    '어케': '어떻게',
    '어캐': '어떻게',

    '걍': '그냥',

    '커엽': '귀엽',
    '존맛': '정말 맛있다',

    '꿀잼': '정말 재미있다',
    '노잼': '재미없다',

    '실화냐': '정말이니',

    'GOAT': '최고',
    'goat': '최고',
}

# phrase-level normalization
PHRASE_RULES = [

    ('개좋', '매우 좋'),
    ('개웃기', '매우 웃기'),
    ('개귀엽', '매우 귀엽'),
    ('개잘', '매우 잘'),

    ('존나 웃기', '매우 웃기'),
    ('존나 잘', '매우 잘'),

    ('ㄹㅇ 개', '정말 매우'),
]

# fillers to remove
FILLERS = [
    'ㅋㅋㅋㅋㅋㅋ',
    'ㅋㅋㅋㅋㅋ',
    'ㅋㅋㅋㅋ',
    'ㅋㅋㅋ',
    'ㅋㅋ',
    'ㅠㅠㅠㅠ',
    'ㅠㅠㅠ',
    'ㅠㅠ',
    'ㄷㄷ',
    'ㄹㅇ',
]

# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def is_valid_comment(text):

    if not isinstance(text, str):
        return False

    text = text.strip()

    if len(text) < 5:
        return False

    if len(text) > 80:
        return False

    if not re.search(r'[가-힣]', text):
        return False

    return True


def clean_text(text):

    text = text.strip()

    # URL 제거
    text = re.sub(r'http\S+', '', text)

    # 공백 정리
    text = re.sub(r'\s+', ' ', text)

    return text


# ═══════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════

def normalize_comment(text):

    norm = text

    # timestamps 제거
    norm = re.sub(
        r'\b\d{1,2}:\d{2}\b',
        '',
        norm
    )

    # fillers 제거
    for filler in FILLERS:
        norm = norm.replace(filler, '')

    # phrase rules
    for noisy, clean in PHRASE_RULES:
        norm = norm.replace(noisy, clean)

    # token rules
    tokens = norm.split()

    normalized_tokens = []

    for token in tokens:

        stripped = token.strip()

        if stripped in SLANG_RULES:
            normalized_tokens.append(
                SLANG_RULES[stripped]
            )
        else:
            normalized_tokens.append(token)

    norm = ' '.join(normalized_tokens)

    # repeated punctuation cleanup
    norm = re.sub(r'[~]{2,}', '~', norm)
    norm = re.sub(r'[!]{2,}', '!', norm)
    norm = re.sub(r'[?]{2,}', '?', norm)
    norm = re.sub(r'[.]{2,}', '.', norm)

    # extra spaces cleanup
    norm = re.sub(r'\s+', ' ', norm)

    norm = norm.strip()

    return norm


# ═══════════════════════════════════════════════════════════════
# Build Dataset
# ═══════════════════════════════════════════════════════════════

def build_dataset(comments, max_samples=1000):

    dataset = []

    seen = set()

    for raw in comments:

        raw = clean_text(raw)

        if not is_valid_comment(raw):
            continue

        norm = normalize_comment(raw)

        # normalization 변화 없는 경우 skip
        if raw == norm:
            continue

        # 너무 짧은 norm 제거
        if len(norm) < 3:
            continue

        pair = (raw, norm)

        if pair in seen:
            continue

        seen.add(pair)

        dataset.append({
            "raw": raw,
            "norm": norm,
            "lang": "ko"
        })

        if len(dataset) >= max_samples:
            break

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
        default='real_yt_normalization.jsonl'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=5000
    )

    args = parser.parse_args()

    # load comments
    with open(args.input, 'r', encoding='utf-8') as f:
        comments = json.load(f)

    print(f'\nloaded comments: {len(comments)}')

    dataset = build_dataset(
        comments,
        max_samples=args.samples
    )

    print(f'\nbuilt dataset: {len(dataset)}')

    # save jsonl
    with open(args.output, 'w', encoding='utf-8') as f:

        for row in dataset:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + '\n'
            )

    print(f'\nsaved -> {args.output}')

    # preview
    print('\nSAMPLE OUTPUTS:\n')

    for x in random.sample(
        dataset,
        min(20, len(dataset))
    ):

        print('RAW :', x['raw'])
        print('NORM:', x['norm'])
        print()


if __name__ == '__main__':
    main()
