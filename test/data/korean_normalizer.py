"""
Korean YouTube Comment Normalizer
Target: MultiLexNorm Korean track

Strategy:
  1. Rule-based: comprehensive slang/abbreviation dictionary
  2. Morpheme-aware: prefix pattern matching (개-, 존나-, etc.)
  3. Context-aware: position-sensitive disambiguation
  4. YouTube-specific: timestamps, @tags, hashtags, emoji removal
  5. Jamo decomposition: systematic filler handling
  6. Suffix normalization: 구어체 어미 교정
  7. Typo correction: 됬→됐 등 표준 오타
"""

import json
import re
import random
import argparse
from typing import Optional

random.seed(42)


# ═══════════════════════════════════════════════════════════════
# 1. SLANG DICTIONARY (token-level exact match)
# ═══════════════════════════════════════════════════════════════

SLANG_DICT = {

    # ── 정도/강조 부사 ──────────────────────────────────────────
    '존나': '매우',
    '존내': '매우',
    '겁나': '매우',
    '겁내': '매우',
    '넘': '너무',
    '너모': '너무',
    '졸라': '매우',
    '쫌': '조금',
    '암튼': '아무튼',
    '아무튼간에': '아무튼',
    '어쨌든간': '어쨌든',
    '하여튼': '아무튼',
    '레알': '정말',
    'ㄹㅇ': '정말',
    '리얼': '정말',
    '실화냐': '정말이야',
    '실화': '정말',
    '솔까':'솔직히',
    '아싸':'아웃사이더',
    '인싸':'인사이더',
    '틀딱':'어르신',
    '한남':'한국 남자',
    '한녀':'한국 여자',
    '맞말': '맞는 말',
    '초딩': '초등학생',
    '중딩': '중학생',
    '고딩': '고등학생',
    '새끼': '친구',
    '겜': '게임',
    '짱깨':'중국',


    # ── 긍정/부정 반응 ──────────────────────────────────────────
    'ㅇㅈ': '인정',
    'ㄴㄴ': '아니',
    'ㄴㄴㄴ': '아니',
    'ㄴㄴㄴㄴ': '아니',
    'ㅇㅇ': '응',
    'ㅇㅇㅇ': '응',
    'ㅇㅇㅇㅇ': '응',
    'ㅇ': '응',
    'ㄱㅊ': '괜찮아',
    'ㄱㅊㄱㅊ': '괜찮아',
    '괜찬': '괜찮아',
    '괜찬아': '괜찮아',
    'ㅁㅊ': '미쳤다',
    'ㅁㅊㄴ': '미쳤나',
    '미침': '미쳤다',
    '미쳐': '미쳤다',
    'ㅅㅂ': '짜증난다',
    '병신': '바보',
    'ㅂㅅ': '바보',
    'ㅄ': '바보',
    'ㅅㄲ': '친구',

    # ── 감사/인사 ────────────────────────────────────────────────
    'ㄱㅅ': '고마워',
    'ㄱㅅㄱㅅ': '고마워',
    'ㄳ': '고마워',
    '고마웡': '고마워',
    '감솨': '감사해',
    '감사합니당': '감사합니다',
    '감사해용': '감사해요',
    '고맙습니당': '고맙습니다',
    '반가웡': '반가워',
    'ㅊㅋ': '축하해',
    'ㅊㅋㅊㅋ': '축하해',

    # ── 감탄/놀람 ────────────────────────────────────────────────
    'ㄷㄷ': '대단해',
    'ㄷㄷㄷ': '대단해',
    'ㅎㄷㄷ': '대단해',
    '헐': '어머',
    '대박': '굉장해',
    '대박이다': '굉장하다',
    '대박이야': '굉장해',
    '쩐다': '굉장하다',
    '쩐디': '굉장해',
    '쩔어': '굉장해',
    '쩔어요': '굉장해요',
    '장난없다': '굉장하다',
    '장난없어': '굉장해',
    'ㅈㄴ': '매우',

    # ── 어떻게/왜/뭐 ────────────────────────────────────────────
    '어케': '어떻게',
    '어캐': '어떻게',
    '왤케': '왜 이렇게',
    '왤캐': '왜 이렇게',
    '왜케': '왜 이렇게',
    '뭔': '무슨',
    '머야': '뭐야',
    '머임': '뭐야',
    '머': '뭐',
    '어디써': '어디서',

    # ── 행동/동작 ────────────────────────────────────────────────
    '걍': '그냥',
    '해봄': '해봤어',
    '했음': '했어',
    '갔음': '갔어',
    '봤음': '봤어',
    '먹었음': '먹었어',
    '들었음': '들었어',
    '됐음': '됐어',
    '됬음': '됐어',
    '됬어': '됐어',
    '됬': '됐',
    '됬다': '됐다',
    '됬는데': '됐는데',

    # ── 오타 교정 ────────────────────────────────────────────────
    '돼서': '되어서',
    '이뻐': '예뻐',
    '이쁜': '예쁜',
    '이쁘다': '예쁘다',
    '이쁘게': '예쁘게',
    '이쁨': '예쁨',

    # ── 묘사/상태 ────────────────────────────────────────────────
    '커엽': '귀여워',
    '귀욥': '귀여워',
    '귀여봐': '귀여워',
    '귀엽당': '귀엽다',
    '귀엽네용': '귀엽네요',
    '예쁘당': '예쁘다',
    '이뻐요': '예뻐요',
    '잘생겼당': '잘생겼다',

    # ── 재미/맛 관련 ────────────────────────────────────────────
    '꿀잼': '재미있어',
    '개꿀잼': '매우 재미있어',
    '노잼': '재미없어',
    '핵노잼': '매우 재미없어',
    '존맛': '정말 맛있어',
    '존맛탱': '정말 맛있어',
    '꿀맛': '매우 맛있어',
    '개맛없': '매우 맛없어',

    # ── 인터넷/밈 ────────────────────────────────────────────────
    'GOAT': '최고',
    'goat': '최고',
    '갓': '최고',
    '레전드': '전설',
    '레전': '전설',
    '팩트': '사실',
    '팩폭': '사실 폭격',
    '현타': '현실감',
    '소름돋': '소름 돋아',
    '소름돋아': '소름 돋아',
    'TMI': '불필요한 정보',
    'tmi': '불필요한 정보',

    # ── 감정 표현 ────────────────────────────────────────────────
    '빡쳐': '화났어',
    '빡침': '화남',
    '열받아': '화났어',
    '열받음': '화났어',
    'ㅡㅡ': '짜증나',
    ';;': '당황스러워',

    # ── 유튜브 특화 ─────────────────────────────────────────────
    'ㅅㄱ': '수고해',
    'ㅅㄱㅅㄱ': '수고해',
    '구독각': '구독해야겠어',
    '구독박고': '구독하고',
    '좋아요박고': '좋아요 누르고',
    '좋아요박': '좋아요 눌러',
    '첫댓': '첫 번째 댓글',

    # ── 줄임말 ──────────────────────────────────────────────────
    '최애': '가장 좋아하는',
    '최애곡': '가장 좋아하는 노래',
    '차애': '두 번째로 좋아하는',
    '취저': '취향 저격',
    '취향저격': '취향 저격',
    '댕댕': '강아지',
    '댕댕이': '강아지',
    '냥이': '고양이',

    # ── 정중체 오타 ──────────────────────────────────────────────
    '안녕하세용': '안녕하세요',
    '맞아용': '맞아요',
    '그렇군용': '그렇군요',
    '그렇죠용': '그렇죠',
    '이에용': '이에요',
    '에용': '에요',
    '해용': '해요',
    '네용': '네요',
    '같아용': '같아요',
    '있어용': '있어요',
    '없어용': '없어요',
    '좋아용': '좋아요',
}

# 사전 키를 길이 내림차순으로 정렬해 longest-match 매칭 보장
# (예: '레전드'가 '레전'보다 먼저 매칭되는 오류 방지)
_SORTED_SLANG_KEYS = sorted(SLANG_DICT.keys(), key=len, reverse=True)


# ═══════════════════════════════════════════════════════════════
# 2. PREFIX RULES (형태소 내부 접두사)
# ═══════════════════════════════════════════════════════════════

# 접두사 규칙: 형용사 어간 whitelist 앞에만 적용 (오적용 방지)
_ADJ_WHITELIST = (
    '좋아|좋은|좋다|웃겨|웃긴|웃기|귀여|슬퍼|슬픈|힘들|빠르|느리|맛있|맛없|예뻐|예쁜|예쁘|별로|재미|많이|크다|작다|높다|낮다|쉽|어렵|무서|싫어|춥|덥'
)
_ADJ_PAT = re.compile(rf'^(?:{_ADJ_WHITELIST})')

PREFIX_RULES = [
    # 존나/존내/겁나 — 단독 강조 부사이므로 그대로 적용
    (re.compile(r'^존나(?=[가-힣]{2,})'), '매우 '),
    (re.compile(r'^존내(?=[가-힣]{2,})'), '매우 '),
    (re.compile(r'^겁나(?=[가-힣]{2,})'), '매우 '),
    # 개/핵 — whitelist 형용사 앞에만 적용
    (re.compile(rf'^개(?={_ADJ_WHITELIST})'), '매우 '),
    (re.compile(rf'^핵(?={_ADJ_WHITELIST})'), '매우 '),
    (re.compile(r'^ㄹㅇ(?=[가-힣]{2,})'), '정말 '),
]


# ═══════════════════════════════════════════════════════════════
# 3. PHRASE-LEVEL RULES
# ═══════════════════════════════════════════════════════════════

PHRASE_RULES = [
    ('개좋아', '매우 좋아'),
    ('개좋은', '매우 좋은'),
    ('개좋다', '매우 좋다'),
    ('개웃겨', '매우 웃겨'),
    ('개웃긴', '매우 웃긴'),
    ('개웃기다', '매우 웃기다'),
    ('개귀여워', '매우 귀여워'),
    ('개귀엽다', '매우 귀엽다'),
    ('개슬퍼', '매우 슬퍼'),
    ('개별로', '매우 별로야'),
    ('개잼', '매우 재미있어'),
    ('개웃기', '매우 웃기'),
    ('개잘', '매우 잘'),
    ('개많이', '매우 많이'),
    ('개힘들', '매우 힘들'),
    ('개빠르', '매우 빠르'),
    ('개느리', '매우 느리'),
    ('개맛있', '매우 맛있'),
    ('개예뻐', '매우 예뻐'),
    ('개이쁘', '매우 예쁘'),

    ('존잘', '정말 잘생겼어'),
    ('존예', '정말 예뻐'),
    ('존귀', '정말 귀여워'),

    ('ㄹㅇ 개', '정말 매우'),
    ('진짜 개', '정말 매우'),
    ('진짜 존나', '정말 매우'),

    ('왤케 좋아', '왜 이렇게 좋아'),
    ('왤케 웃겨', '왜 이렇게 웃겨'),
    ('왤케 예뻐', '왜 이렇게 예뻐'),
    ('왤케 잘해', '왜 이렇게 잘해'),
    ('왤케 이뻐', '왜 이렇게 예뻐'),
]


# ═══════════════════════════════════════════════════════════════
# 4. YOUTUBE-SPECIFIC CLEANUP PATTERNS
# ═══════════════════════════════════════════════════════════════

YOUTUBE_PATTERNS = [
    (re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\b'), ''),       # 타임스탬프
    (re.compile(r'@\w+'), ''),                                  # @멘션
    (re.compile(r'#\S+'), ''),                                  # #해시태그
    (re.compile(r'https?://\S+'), ''),                          # URL
    (re.compile(r'www\.\S+'), ''),                              # URL
    (re.compile(
        r'[\U0001F300-\U0001FAFF\U00002600-\U000027BF'
        r'\U0001F000-\U0001F9FF\U0001FA00-\U0001FA6F'
        r'\U0001FA70-\U0001FAFF\U00002702-\U000027B0]+'
    ), ''),    # 이모지
]


# ═══════════════════════════════════════════════════════════════
# 5. TYPO CORRECTION (정규표현식 기반)
# ═══════════════════════════════════════════════════════════════

# 자모 suffix 제거 패턴 (한글 뒤에 붙은 의미없는 자모)
JAMO_SUFFIX_PATTERN = re.compile(r'(?<=[가-힣])[ㄱ-ㅎ]+$')

# 한글 뒤에 붙은 ㅠ/ㅜ 제거 (예: 좋다ㅠㅠ → 좋다)
JAMO_VOWEL_SUFFIX_PATTERN = re.compile(r'(?<=[가-힣])[ㅠㅜ]+')

TYPO_PATTERNS = [
    # 됬 → 됐
    (re.compile(r'됬'), '됐'),
    # 이뻐/이쁘 → 예뻐/예쁘
    (re.compile(r'이뻐'), '예뻐'),
    (re.compile(r'이쁘'), '예쁘'),
    (re.compile(r'이쁜'), '예쁜'),
    # 구어체 '에용/해용/네용' 등 어미 교정 (토큰 끝)
    (re.compile(r'이에용\b'), '이에요'),
    (re.compile(r'에용\b'), '에요'),
    (re.compile(r'해용\b'), '해요'),
    (re.compile(r'네용\b'), '네요'),
    (re.compile(r'죠용\b'), '죠'),
    (re.compile(r'이용\b'), '이요'),
    (re.compile(r'욤\b'), '요'),
    (re.compile(r'염\b'), '요'),
    # '당' 어미: 한글 뒤 문장 끝 '당' → '다'
    (re.compile(r'(?<=[가-힣])당\b'), '다'),

    # ── 쌍자음 → 단자음 오류 교정 ───────────────────────────────
    (re.compile(r'잇어'), '있어'),
    (re.compile(r'잇는'), '있는'),
    (re.compile(r'잇다'), '있다'),
    (re.compile(r'잇음'), '있음'),
    (re.compile(r'업어'), '없어'),
    (re.compile(r'업다'), '없다'),
    (re.compile(r'업는'), '없는'),
    (re.compile(r'업음'), '없음'),
    (re.compile(r'\b읍어\b'), '없어'),
    (re.compile(r'\b안자\b'), '앉아'),
    (re.compile(r'\b안자서\b'), '앉아서'),
    (re.compile(r'\b몰아\b'), '몰라'),
    (re.compile(r'\b알아서해\b'), '알아서 해'),
    (re.compile(r'\b할수잇어\b'), '할 수 있어'),
    (re.compile(r'\b할수업어\b'), '할 수 없어'),
]


# ═══════════════════════════════════════════════════════════════
# 6. CONTEXT-AWARE TOKEN DISAMBIGUATION
# ═══════════════════════════════════════════════════════════════

def disambiguate_token(
    token: str,
    prev_token: Optional[str],
    next_token: Optional[str]
) -> str:

    # 자모 필러: 단독 ㅠ+/ㅜ+ 토큰 → 제거
    if re.fullmatch(r'[ㅠㅜ]+', token):
        return ''
    if re.fullmatch(r'ㄷ+', token):
        if token in SLANG_DICT:
            return SLANG_DICT[token]
        return ''

    # '개' 단독 토큰: whitelist 형용사 앞에만 '매우'
    if token == '개':
        if next_token and _ADJ_PAT.search(next_token):
            return '매우'
        return token

    # '진짜/진쨔' 단독 토큰 → '정말'
    if token in ('진짜', '진쨔', '진짜로'):
        return '정말' if token != '진짜로' else '정말로'

    # 사전 조회 (exact match)
    if token in SLANG_DICT:
        return SLANG_DICT[token]

    # 복합 토큰: longest-match로 접두 슬랭 처리
    # BUG FIX: _SORTED_SLANG_KEYS(길이 내림차순)로 순회해
    #          '레전드'가 '레전'보다 먼저 매칭되는 오류 방지
    for key in _SORTED_SLANG_KEYS:
        if token.startswith(key) and len(key) < len(token):
            remainder = token[len(key):]
            is_jamo_key = re.fullmatch(r'[ㄱ-ㅎㅏ-ㅣ]+', key)
            # BUG FIX: remainder 1글자도 허용 (개꿀잼임 등)
            if is_jamo_key or len(remainder) >= 1:
                return SLANG_DICT[key] + remainder

    return token


# ═══════════════════════════════════════════════════════════════
# 7. PREFIX MORPHEME PROCESSING
# ═══════════════════════════════════════════════════════════════

def apply_prefix_rules(token: str) -> str:
    for pattern, replacement in PREFIX_RULES:
        match = pattern.match(token)
        if match:
            rest = token[match.end():]
            return replacement + rest
    return token


def strip_jamo_suffix(token: str) -> str:
    # 한글 뒤 자음 자모 제거
    token = JAMO_SUFFIX_PATTERN.sub('', token)
    # 한글 뒤 모음 자모(ㅠ/ㅜ) 제거
    token = JAMO_VOWEL_SUFFIX_PATTERN.sub('', token)
    return token


# ═══════════════════════════════════════════════════════════════
# 8. VALIDATION
# ═══════════════════════════════════════════════════════════════

def is_valid_comment(text: str) -> bool:
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < 5 or len(text) > 120:
        return False
    if not re.search(r'[가-힣]', text):
        return False
    # 의미없는 자모만으로 구성
    if re.fullmatch(r'[ㄱ-ㅎㅏ-ㅣ\s!?~.]+', text):
        return False
    return True


# ═══════════════════════════════════════════════════════════════
# 9. TEXT CLEANING
# ═══════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    text = text.strip()
    for pattern, replacement in YOUTUBE_PATTERNS:
        text = pattern.sub(replacement, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ═══════════════════════════════════════════════════════════════
# 10. MAIN NORMALIZATION PIPELINE
# ═══════════════════════════════════════════════════════════════

def normalize_comment(text: str) -> str:
    norm = text

    # Step 0: YouTube-specific cleanup (timestamps, @tags, etc.)
    for pattern, replacement in YOUTUBE_PATTERNS:
        norm = pattern.sub(replacement, norm)

    # Step 1: Phrase-level rules (토큰화 전)
    for noisy, clean in PHRASE_RULES:
        norm = norm.replace(noisy, clean)

    # Step 2: Typo correction (regex 기반)
    for pattern, replacement in TYPO_PATTERNS:
        norm = pattern.sub(replacement, norm)

    # Step 3: Prefix morpheme rules + trailing jamo cleanup
    tokens = norm.split()
    tokens = [apply_prefix_rules(t) for t in tokens]
    tokens = [strip_jamo_suffix(t) for t in tokens]
    norm = ' '.join(tokens)

    # Step 4: Context-aware token normalization
    tokens = norm.split()
    result_tokens = []
    for i, token in enumerate(tokens):
        prev_tok = tokens[i - 1] if i > 0 else None
        next_tok = tokens[i + 1] if i < len(tokens) - 1 else None
        result_tokens.append(
            disambiguate_token(token, prev_tok, next_tok)
        )
    norm = ' '.join(t for t in result_tokens if t)

    # Step 5: Remove residual jamo fillers
    norm = re.sub(r'ㄷ{3,}', '', norm)

    # Step 6: Final whitespace cleanup
    norm = re.sub(r'\s+', ' ', norm).strip()

    return norm


# ═══════════════════════════════════════════════════════════════
# 11. DATASET BUILDER
# ═══════════════════════════════════════════════════════════════

def build_dataset(comments: list, max_samples: int = 5000) -> list:
    dataset = []
    seen = set()

    for raw in comments:
        raw = clean_text(raw)

        if not is_valid_comment(raw):
            continue

        norm = normalize_comment(raw)

        if raw == norm:
            continue

        if len(norm.strip()) < 3:
            continue

        # 과도한 삭제 방지 (원문의 30% 미만으로 줄어들면 skip)
        if len(norm) < len(raw) * 0.3:
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
# 12. SELF-TEST
# ═══════════════════════════════════════════════════════════════

TEST_CASES = [
    # (raw, expected_norm)
    ("개웃긴데ㅋㅋㅋ 진짜", "매우 웃긴데 정말"),
    ("ㄹㅇ 존나 좋다ㅠㅠ", "정말 매우 좋다"),
    ("왤케 이뻐요 진짜 레알", "왜 이렇게 예뻐요 정말 정말"),
    ("됬다 이제 다왔네 ㄱㅊ", "됐다 이제 다왔네 괜찮아"),
    ("귀엽당ㅠㅠ 진짜 개귀여워", "귀엽다 정말 매우 귀여워"),
    ("1:23 여기서 소름ㄷㄷ 대박이다", "여기서 소름 굉장하다"),
    ("구독각이다 ㄱㅅ", "구독해야겠어이다 고마워"),
    ("ㄷㄷ 존나 쩐다 레전드네", "대단해 매우 굉장하다 전설네"),
    ("꿀잼ㅋㅋ 개웃겨 ㄹㅇ", "재미있어 매우 웃겨 정말"),
    ("존맛탱이에용 또 먹고싶다", "정말 맛있어이에요 또 먹고싶다"),
    ("ㅇㅇ 맞아 개꿀잼임", "응 맞아 매우 재미있어임"),
    ("감사해용 진짜 최애곡이에용", "감사해요 정말 가장 좋아하는 노래이에요"),
]

def run_tests():
    print("\n" + "="*65)
    print("SELF-TEST RESULTS")
    print("="*65)
    passed = 0
    for raw, expected in TEST_CASES:
        result = normalize_comment(raw)
        ok = result == expected
        if ok:
            passed += 1
        status = "✓" if ok else "✗"
        print(f"{status} RAW:      {raw}")
        print(f"  RESULT:   {result}")
        if not ok:
            print(f"  EXPECTED: {expected}")
        print()
    print(f"Passed: {passed}/{len(TEST_CASES)}")
    print("="*65 + "\n")


# ═══════════════════════════════════════════════════════════════
# 13. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Korean YouTube Comment Normalizer for MultiLexNorm"
    )
    parser.add_argument('--input', type=str, default='youtube_comments.json')
    parser.add_argument('--output', type=str, default='real_yt_normalization.jsonl')
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--test', action='store_true', help='Run self-tests')
    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    with open(args.input, 'r', encoding='utf-8') as f:
        comments = json.load(f)
    print(f'\nLoaded comments: {len(comments)}')

    dataset = build_dataset(comments, max_samples=args.samples)
    print(f'Built dataset:   {len(dataset)}')

    with open(args.output, 'w', encoding='utf-8') as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Saved → {args.output}')

    print('\n── SAMPLE OUTPUTS ──────────────────────────────────\n')
    for x in random.sample(dataset, min(20, len(dataset))):
        print(f'RAW : {x["raw"]}')
        print(f'NORM: {x["norm"]}')
        print()


if __name__ == '__main__':
    main()
