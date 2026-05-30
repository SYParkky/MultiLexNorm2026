# MFR-Hybrid Post-Processing: Technical Summary
*for team members*

---

## Background & Motivation

After training grouped ByT5-base models per language cluster, we analyzed per-language validation statistics and identified a systematic **over-normalization problem** in specific languages:

| Language | ERR | TP | FP | FN |
|---|---|---|---|---|
| th | 10.81 | 489 | **404** | 297 |
| ja | 15.23 | 252 | **148** | 431 |
| ko | 3.61 | 31 | 25 | 135 |

Thai is the most critical case: **FP (404) exceeds FN (297)**, meaning the model incorrectly modifies more standard tokens than it fails to normalize. Since ERR = (TP − FP) / (TP + FN), high FP directly penalizes the score.

### Why does this happen?

| Language | Root Cause |
|---|---|
| **Thai** | No word boundaries + UTF-8 byte inefficiency in ByT5 (Xue et al., 2022) + limited training data |
| **Japanese** | Three mixed scripts (hiragana, katakana, kanji) → ambiguous standard/non-standard boundaries |
| **Korean** | Very small training data → model hasn't learned reliable patterns |

The fundamental solution would be an LLM-based pipeline (Buaphet et al., 2026), but this falls outside the scope of the assignment. Instead, we propose an **inference-time post-processing method**.

---

## Method

> No additional training required. Applied only at inference time, after the model generates a prediction.

### Stage 1 — MFR Consistency Filter

We build an MFR (Most-Frequent-Replacement) dictionary from the training data, following the same definition as the official baseline (van der Goot et al., 2021). For each word *w*, we additionally compute a **consistency ratio**:

$$\rho(w) = \frac{\max_n \, \text{count}(w \to n)}{\sum_n \, \text{count}(w \to n)}$$

If the model predicts a change but either:
- **MFR(*w*) = *w*** — the word is most often left unchanged in training data, or
- **ρ(*w*) < τ_m** — the word has inconsistent normalization patterns,

→ the prediction is **overridden** and the original word is retained.

### Stage 2 — Confidence Thresholding *(available but found ineffective for ByT5)*

We compute the maximum softmax probability over the vocabulary at the first decoding step as a confidence proxy — the Softmax Response (SR) measure from Geifman & El-Yaniv (2017):

$$\text{conf}(w) = \max_v \, P(v \mid w)$$

If conf(*w*) < τ_c, the original word is retained.

> **⚠️ Empirical finding:** ByT5's byte-level generation produces consistently high first-token confidence scores, making this filter largely ineffective in practice. **MFR filter only is used in the final system.**

### Final Decision Rule

$$\hat{w} = \begin{cases} w & \text{if MFR}(w) = w \text{ or } \rho(w) < \tau_m \\ \text{model}(w) & \text{otherwise} \end{cases}$$

### Selective Application

Post-processing is applied **only to th / ja / ko**.  
Germanic, Slavic, and SEA groups already achieve high ERR — applying the filter degrades their performance by suppressing true positives.

| Group | Post-processing |
|---|---|
| th, ja, ko | ✅ MFR filter ON (τ_m = 0.3) |
| germanic, slav, sea | ❌ OFF |

---

## Results

| Language | ERR (before) | ERR (after) | Change |
|---|---|---|---|
| th | 10.81 | **38.42** | +27.61 |
| ja | 15.23 | **18.89** | +3.66 |
| ko | 3.61 | **10.84** | +7.23 |
| germanic | 31.47 | 31.47 | unchanged |
| slav | 53.80 | 53.80 | unchanged |
| sea | 74.12 | 74.12 | unchanged |
| **overall** | **51.76** | **TBD** | *(selective run pending)* |

---

## Key Insight

The confidence threshold (Stage 2) was found to be **ineffective for ByT5** because byte-level models produce high-confidence first tokens structurally. This is a novel empirical finding worth noting in the report. The **MFR consistency filter (Stage 1) is the effective component**.

---

## References

- van der Goot, R. et al. (2021). MultiLexNorm: A Shared Task on Multilingual Lexical Normalization. *W-NUT 2021*, pp. 171–182. https://aclanthology.org/2021.wnut-1.38
- Samuel, D. & Straka, M. (2021). ÚFAL at MultiLexNorm 2021: Improving Multilingual Lexical Normalization by Fine-tuning ByT5. *W-NUT 2021*, pp. 483–492. https://aclanthology.org/2021.wnut-1.54
- Xue, L. et al. (2022). ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models. *TACL*, vol. 10, pp. 291–306. https://aclanthology.org/2022.tacl-1.17
- Geifman, Y. & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. *NeurIPS 2017*, pp. 4878–4887. https://arxiv.org/abs/1705.08500
- Buaphet, W. et al. (2026). MultiLexNorm++. arXiv:2601.16623. https://arxiv.org/abs/2601.16623
