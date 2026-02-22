# Experiment Results

## Model Architecture

- **Encoder:** MedSigLIP-448 vision encoder (~429M params, frozen)
- **Head:** LayerNorm → Dropout(0.3) → Linear(embed_dim, 512) → GELU → Dropout(0.3) → Linear(512, 7)
- **Trainable parameters:** 596K
- **Input resolution:** 448x448

## Experiment 1: HAM10000 Baseline

**Training:** HAM10000 only, Focal Loss (gamma=2.0), 2x melanoma class weight, cosine LR schedule.

### HAM10000 Validation (In-Domain)

| Metric | Value |
|--------|------:|
| Accuracy | 75.5% |
| Balanced Accuracy | **78.9%** |
| Macro F1 | 72.2% |
| Melanoma Recall | **85.7%** |
| Melanoma Precision | 35.4% |

### Per-Class Recall (HAM10000)

| Class | Recall |
|-------|-------:|
| akiec | 75.4% |
| bcc | 87.4% |
| bkl | 70.0% |
| df | 78.3% |
| mel | **85.7%** |
| nv | 73.6% |
| vasc | 82.1% |

### PAD-UFES-20 Validation (Cross-Domain)

| Metric | Value |
|--------|------:|
| Accuracy | 43.8% |
| Balanced Accuracy | 49.9% |
| Macro F1 | 31.2% |
| Melanoma Recall | 54.5% |

**Key finding:** ~30% performance drop from dermoscopic to clinical images, confirming significant domain gap.

---

## Experiment 2: Mixed Training (HAM10000 + PAD-UFES-20)

**Training:** Combined datasets with 3x PAD-UFES oversampling, Focal Loss, 2x melanoma weight, stronger augmentation (ColorJitter 0.3, GaussianBlur).

### Comparison: Experiment 1 vs Experiment 2

| Dataset | Metric | Exp 1 (HAM only) | Exp 2 (Mixed) | Delta |
|---------|--------|:-----------------:|:-------------:|:-----:|
| HAM10000 | Balanced Accuracy | **78.9%** | 69.6% | -9.3 |
| HAM10000 | Melanoma Recall | **85.7%** | 77.6% | -8.1 |
| PAD-UFES-20 | Balanced Accuracy | 49.9% | **81.8%** | **+31.9** |
| PAD-UFES-20 | Melanoma Recall | 54.5% | **81.8%** | **+27.3** |

### Fairness Analysis (PAD-UFES-20, Fitzpatrick Skin Types)

| Fitzpatrick Group | n | Exp 1 Accuracy | Exp 2 Accuracy | Exp 1 Mel Recall | Exp 2 Mel Recall |
|:-----------------:|:-:|:--------------:|:--------------:|:----------------:|:----------------:|
| I-II | 168 | 51.2% | 86.3% | 50.0% | 75.0% |
| III-IV | 93 | 46.2% | 82.8% | 66.7% | **100.0%** |
| V-VI | 0 | — | — | — | — |
| Unknown | 159 | 35.2% | 81.1% | 0.0% | 0.0% |

### Analysis

1. **Cross-domain improvement exceeded optimistic predictions.** We expected +10-15% on PAD-UFES-20; actual improvement was +31.9%.

2. **HAM10000 trade-off is acceptable.** The -9.3% drop on dermoscopic images is expected. The target use case is primary care smartphone triage, where PAD-UFES-20 performance is the relevant benchmark.

3. **Fairness improved.** Fitzpatrick III-IV melanoma recall went from 66.7% to 100%. The gap between I-II (75%) and III-IV (100%) inverts the typical dermatology AI bias pattern.

4. **Limitation:** The "Unknown" Fitzpatrick group shows 0% melanoma recall in both experiments. This needs further investigation.

5. **Limitation:** No Fitzpatrick V-VI samples available in either dataset.

---

## Summary Table

| Model | Training Data | HAM10K Bal Acc | HAM10K Mel Recall | PAD Bal Acc | PAD Mel Recall | Recommended For |
|:-----:|:-------------:|:--------------:|:-----------------:|:-----------:|:--------------:|:---------------:|
| M1 | HAM10000 | **78.9%** | **85.7%** | 49.9% | 54.5% | Dermoscopic imaging |
| M2 | HAM + PAD | 69.6% | 77.6% | **81.8%** | **81.8%** | Clinical/smartphone triage |

---

## References

1. Tschandl P, et al. "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions." *Sci Data.* 2018.
2. Pacheco AG, et al. "PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones." *Data Brief.* 2020.
3. Daneshjou R, et al. "Disparities in dermatology AI performance on a diverse, curated clinical image set." *Sci Adv.* 2022.
