# augment_wiki.py — Wikipedia Corruption Augmentation

Generates synthetic `(noisy input → clean target)` training pairs from Wikipedia text.

---

## Quick Start

### 1. Install dependencies
```bash
pip install datasets konlpy fugashi unidic-lite
```

KoNLPy requires Java — check if you have it:
```bash
java -version
```
If not:
```bash
brew install java   # macOS
```

### 2. Run
```bash
# Default: ko + ja + en + de, 5000 pairs each
python data/augment_wiki.py

# Custom languages and sample size
python data/augment_wiki.py --langs ko en --samples 3000 --output data/wiki_aug.jsonl
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--langs` | `ko ja en de` | Languages to generate |
| `--samples` | `5000` | Pairs per language |
| `--variants` | `3` | Corruption variants per word |
| `--output` | `data/wiki_aug.jsonl` | Output file path |

### 3. Run training
 
`finetune_byt5-base-wiki.py` already includes wiki augmentation — no manual changes needed.
 
Just make sure `wiki_aug.jsonl` exists before training, then clear the cache and run:
 
```bash
# Clear cache so it re-tokenizes with augmented data
rm -rf ./data/cached/
 
# Train
python train/finetune_byt5-base-wiki.py
```
 
> ⚠️ If `wiki_aug.jsonl` is not found, training will continue with the original data only and print a warning.
 

---

## What it does

Streams clean Wikipedia sentences → applies corruption rules → saves `(noisy, clean)` pairs.

The **target is never modified** — only the input is corrupted.

```
Wikipedia text:  "많이"
                    ↓  corruption applied
Output pair:     raw:  "많이ㅋ"    ← noisy input (what model receives)
                 norm: "많이"      ← clean target (what model should output, unchanged)
```

Output is saved as JSONL:
```json
{"raw": "많이ㅋ",    "norm": "많이",    "lang": "ko", "corruption": "filler"}
{"raw": "becuase",   "norm": "because", "lang": "en", "corruption": "swap_chars"}
{"raw": "アリガトウ", "norm": "ありがとう", "lang": "ja", "corruption": "kana_swap"}
```

---

## Implementations
 
### 1. Weighted Corruption Probabilities
 
Each corruption rule has a different probability weight — common real-world errors appear more often than rare ones.
 
 
Example weights for Korean:
 
| Rule | Weight | Reason |
|------|--------|--------|
| filler (`ㅋ/ㅠ/ㅜ/ㅎ/~/...`) | 25% | Very common in Korean online text |
| 자모 decomposition | 20% | Common typo pattern |
| spacing error | 20% | Very common in online writing |
| trail 자모 (`나비ㅣ`, `나방ㅇ`) | 20% | Common informal typing pattern |
| number sub (`ㅇ→0`) | 10% | Moderate frequency |
| word repeat | 5% | Rare |
 
---
 
### 2. Spacing Corruption
 
Simulates missing or incorrect spaces — especially important for Korean, where spacing errors are extremely common in online writing.
 
```
Original:  "자연어"
Corrupted: "자연"       ← one character dropped (spacing collapsed)
 
Original:  "a lot"
Corrupted: "alot"       ← space removed
```
 
---
 
### 3. Realistic Typo Simulation
 
Replaced unrealistic augmentations (e.g. full word duplication) with natural human typing mistakes.
 
**Character deletion** — accidentally skipping a key:
```
"because"  →  "becuse"
"really"   →  "realy"
```
 
**Adjacent character swap** — hitting two keys in the wrong order:
```
"the"   →  "hte"
"from"  →  "form"
```
 
**Vowel repetition** — holding a key too long:
```
"so"      →  "sooo"
"really"  →  "reeeally"
```
 
---
 
### 4. Language-Specific Corruption Rules
 
Each language has rules that reflect how people actually write informally online.
 
**Korean (`ko`)**
```
"많이"  →  "많이ㅋ"     # filler appended (ㅋ/ㅠ/ㅜ/ㅎ/~/...)
"나비"  →  "나비ㅣ"     # trail jamo — last vowel appended
"나방"  →  "나방ㅇ"     # trail jamo — last consonant appended
"이거"  →  "이0거"      # ㅇ replaced with 0 (visual similarity)
"많이"  →  "많ㅏㅇㅣ"   # last syllable decomposed into jamo
```
 
**Japanese (`ja`)**
```
"ありがとう"  →  "アリガトウ"    # hiragana → katakana swap
"ありがとう"  →  "ありーがとう"   # long vowel ー inserted
"ています"   →  "てる"          # colloquial contraction
```
 
**English (`en`)**
```
"going"    →  "goin'"     # drop trailing g (-ing → -in')
"want to"  →  "wanna"     # contraction
"really"   →  "reeeally"  # vowel repetition
```
 
**German (`de`)**
```
"können"  →  "koennen"  # umlaut removed (ü → ue)
"Haus"    →  "haus"     # lowercase (casual writing)
"eine"    →  "ein"      # drop final e
```
 
---
 
### 5. Morpheme-Level Tokenization
 
Instead of splitting by whitespace (which gives poor results for Korean and Japanese), the script uses proper tokenizers.
 
| Language | Tokenizer | Fallback |
|----------|-----------|---------|
| Korean | KoNLPy / Okt | whitespace |
| Japanese | fugashi + unidic-lite | whitespace |
| English | whitespace | — |
| German | whitespace | — |
 
```python
# Korean example
Okt().morphs("자연어처리")
# → ["자연어", "처리"]   ← correct morpheme boundaries
 
"자연어처리".split()
# → ["자연어처리"]       ← treats whole word as one token
```
 
If KoNLPy or fugashi is not installed, the script automatically falls back to whitespace tokenization with a warning — it won't crash.
 
---
 
### 6. Corruption Metadata
 
Every output pair records which corruption rule was applied.
 
```json
{"raw": "becuase", "norm": "because", "lang": "en", "corruption": "swap_chars"}
```
 
This enables:
- Per-rule ablation studies (which rules actually help?)
- Error analysis (which corruptions are hardest for the model?)
- Filtering specific rule types for experiments
---

I didn't add the weights for English, German, and Japanese yet! I think the weights might be adjusted
