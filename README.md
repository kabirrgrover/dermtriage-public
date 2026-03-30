# DermTriage: AI-Powered Skin Lesion Triage for Primary Care

A clinical decision support system that helps primary care physicians identify potentially dangerous skin lesions using smartphone photos. Built on Google's MedSigLIP and MedGemma medical AI foundation models, with explicit fairness evaluation across Fitzpatrick I-VI skin tones.

**Status:** Research prototype. Actively developing binary safety gate with Fitzpatrick17k for improved specificity.

[![Open Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kabirrgrover/dermtriage-public/blob/master/notebooks/03_demo_pipeline.ipynb) [![Open in HF Spaces](https://img.shields.io/badge/HuggingFace-Spaces-blue)](https://huggingface.co/spaces/Kabirgrover/dermtriage)

---

## The Problem

Skin cancer is the most common cancer in the United States, affecting over 5 million Americans annually. Melanoma, the deadliest form, kills approximately 8,000 people each year. Early detection is critical -- the five-year survival rate for melanoma caught early exceeds 99%, but drops to around 30% once the cancer has spread.

Two interconnected problems make early detection difficult:

**1. Access Gap.** Primary care physicians are often the first to see patients with skin concerns, but they achieve only about 45% diagnostic accuracy for melanoma compared to 97% for dermatologists. Dermatologists are in short supply, particularly in rural and underserved communities. Patients in these areas may wait weeks or months for specialist appointments, during which time a cancer can progress significantly.

**2. Fairness Gap.** Most AI tools for skin cancer detection are trained on datasets composed predominantly of lighter skin tones (Fitzpatrick I-III). As a result, these tools perform significantly worse on patients with darker skin. This compounds existing health disparities, as skin cancer on darker skin is already harder to detect visually.

DermTriage addresses both problems by building an AI tool that works on clinical smartphone images and is explicitly trained and evaluated to perform well across different skin tones.

---

## Architecture

DermTriage uses a staged pipeline built on two frozen vision encoders and a trainable classification head:

| Component | Model | Purpose | Parameters |
|-----------|-------|---------|------------|
| Vision Encoder 1 | MedSigLIP-448 | Medical image features (1152D) | ~429M (frozen) |
| Vision Encoder 2 | DermLIP-PanDerm | Dermatology-specific features (512D) | ~86M (frozen) |
| Classification Head | Trainable MLP | 7-class skin lesion classifier | ~596K (trainable) |
| Explainability | Grad-CAM | Visual attention heatmaps | -- |
| Clinical Reasoning | MedGemma-4B | Natural language explanations | 4B |

The vision encoders are frozen during training. Only the classification head is trained, preventing overfitting on relatively small medical datasets while leveraging pretrained medical image features.

### Seven-Class Classification

| Class | Description | Risk Level |
|-------|-------------|------------|
| mel | Melanoma | URGENT -- Refer within 48 hours |
| bcc | Basal cell carcinoma | HIGH -- Refer within 2 weeks |
| akiec | Actinic keratosis / Bowen's disease | MODERATE -- Refer within 2-4 weeks |
| bkl | Benign keratosis | LOW -- Routine monitoring |
| df | Dermatofibroma | LOW -- Routine monitoring |
| nv | Melanocytic nevus | LOW -- Routine monitoring |
| vasc | Vascular lesion | LOW -- Routine monitoring |

### Safety-First Design

In our demo, a basal cell carcinoma image was classified as melanoma (URGENT) rather than BCC (HIGH). This is a feature, not a bug -- both are cancers requiring referral, and the system errs toward the more dangerous diagnosis. Combined with uncertainty flagging, this triggers the correct clinical action: urgent dermatology referral.

---

## Datasets

### Training Datasets

| Dataset | Domain | Images | Fitzpatrick | Role |
|---------|--------|--------|-------------|------|
| HAM10000 | Dermoscopic | 10,015 | Predominantly I-III | Core training (7-class) |
| PAD-UFES-20 | Clinical smartphone | ~2,100 | I-V (Brazilian) | Cross-domain bridge |

### Evaluation Dataset

| Dataset | Domain | Images | Fitzpatrick | Role |
|---------|--------|--------|-------------|------|
| DDI | Clinical photography | 656 | I-VI (balanced) | Fairness evaluation (never used for training) |

DDI (Diverse Dermatology Images) is the primary fairness benchmark. It contains 171 malignant and 485 benign cases with balanced representation across Fitzpatrick I-VI, including 78 unique disease conditions (14 in-distribution, 64 out-of-distribution relative to training classes).

---

## Results

### Important Note on Data Leakage

M2 metrics reported prior to March 2026 used naive random splits on HAM10000, which contained 39.5% data leakage (same lesions appearing in both train and validation sets). All results below labeled "honest" use lesion-aware splits (GroupShuffleSplit on lesion_id). This reduces reported sensitivity from 90.6% to 84.8% but represents honest, reproducible performance.

### Cross-Domain Training

Two experiments evaluate single-domain vs. mixed-domain training:

| Model | Training Data | HAM10K Bal Acc | HAM10K Mel Recall | PAD-UFES Bal Acc | PAD-UFES Mel Recall |
|-------|---------------|:--------------:|:-----------------:|:----------------:|:-------------------:|
| M1 (Baseline) | HAM10000 | **78.9%** | **85.7%** | 49.9% | 54.5% |
| M2 (Mixed) | HAM10000 + PAD-UFES-20 | 69.6% | 77.6% | **81.8%** | **81.8%** |

**M2 is the recommended model** for primary care smartphone triage: it trades 9.3 points of in-domain accuracy for **31.9 points of cross-domain accuracy** on clinical images.

### DDI Fairness Evaluation (Honest Baselines)

Evaluated on DDI with binary malignant/benign framing. All numbers use lesion-aware splits.

| Model | AUC | Sensitivity | Specificity | Referral Rate | V-VI Sensitivity |
|-------|-----|-------------|-------------|---------------|------------------|
| MedSigLIP (honest) | 0.734 | 84.8% | 41.6% | 65.2% | 79.2% |
| DermLIP (honest) | 0.733 | 80.7% | 50.9% | 57.3% | 70.8% |
| Fusion (honest) | 0.742 | 83.6% | 44.1% | 63.1% | 77.1% |
| Ensemble alpha=0.6 | **0.748** | **86.0%** | 44.5% | 63.4% | 79.2% |

### DDI Per-Fitzpatrick Breakdown

| Fitzpatrick | n | AUC | Sensitivity | Specificity |
|-------------|---|-----|-------------|-------------|
| I-II | 208 | 0.701 | 83.7% | 32.7% |
| III-IV | 241 | 0.814 | 87.8% | 51.5% |
| V-VI | 207 | 0.696 | 77.1% | 47.8% |

Sensitivity holds at 77-88% across all Fitzpatrick groups. The fairness gap (max AUC difference across groups) is 11.7 points, within the 15-point target.

### Key Challenge: OOD Over-Referral

Approximately 56% of false positives on DDI come from out-of-distribution conditions (psoriasis, eczema, warts, fungal infections) that the model has never seen during training. These conditions are classified as potentially malignant because the model lacks any representation of non-neoplastic skin conditions. This is a data problem, not an architecture problem, and is the primary motivation for the binary gate work described below.

---

## Experiment History

| Exp | Description | Outcome |
|-----|-------------|---------|
| 1 | HAM10000 baseline | 85.7% mel recall (dermoscopic) |
| 2 | Mixed training (HAM + PAD) | 81.8% mel recall (clinical), 87% domain gap reduction |
| 3 | 8-class model with Fitzpatrick17k "other" class | Rejected -- sensitivity crashed 90.6% to 75.4%, V-VI 60.4% |
| 4 | Energy + Mahalanobis OOD detection | Failed -- no ID/OOD separation in MedSigLIP features |
| 5 | DermLIP encoder swap + NegLabel OOD | DermLIP complements MedSigLIP but lower sensitivity alone |
| 6 | Feature fusion (MedSigLIP + DermLIP) | Does not meet targets. Best AUC=0.748, referral rate 63% |
| 10 | Binary safety gate with SCIN | Failed -- SCIN domain mismatch contaminated decision boundary |
| 10b | Binary ablation (no SCIN) | Binary HAM+PAD matches 7-class (AUC 0.756). Confirmed SCIN was the problem |
| 10c | Binary gate with Fitzpatrick17k | In progress -- domain-aligned OOD data, 114 conditions |

### What the Experiments Taught Us

**Architecture is not the bottleneck.** Feature fusion and encoder swaps showed marginal AUC gains but never met sensitivity or referral targets. The root cause is that the model has never seen the conditions causing most false positives.

**Data domain alignment matters more than volume.** SCIN provided 10K images but its smartphone self-photo domain was too far from clinical photography (cosine similarity 0.60 to DDI). It taught the model "SCIN-style = benign," misclassifying DDI malignancies. Binary training itself works fine -- the ablation (Exp 10b) proved that removing SCIN restored performance to 7-class baseline levels.

**Binary framing avoids probability mass absorption.** Fitzpatrick17k failed as an 8th "other" class (Exp 3) because the "other" class absorbed probability mass from both malignant and benign classes. A binary gate (malignant vs. benign) sidesteps this entirely.

---

## Current Work

**Binary Gate with Fitzpatrick17k (Exp 10c)** -- Training a binary malignant/benign classifier using Fitzpatrick17k (~16.5K clinical photos, 114 dermatological conditions, Fitzpatrick skin type labels) as domain-aligned OOD benign data. The hypothesis: Fitzpatrick17k shares DDI's clinical photography domain and covers the OOD conditions causing the majority of false positives.

If Exp 10c improves specificity, the next step is pipeline integration (binary gate + 7-class diagnostic head + uncertainty routing into a three-zone system: monitor / uncertain / refer). If frozen features hit a ceiling, LoRA fine-tuning with Fitzpatrick17k is the fallback.

---

## Quick Start

### Run the demo (recommended)

1. Open the [demo notebook](https://colab.research.google.com/github/kabirrgrover/dermtriage-public/blob/master/notebooks/03_demo_pipeline.ipynb) in Colab
2. Add your secrets in Colab (Settings > Secrets): `HF_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`
3. Run all cells -- the checkpoint downloads automatically from Google Drive

### Local setup

```bash
git clone https://github.com/kabirrgrover/dermtriage-public.git
cd dermtriage-public
pip install -r requirements.txt
```

```python
from src.pipeline import run_dermtriage_pipeline

report, result = run_dermtriage_pipeline(
    image_path="path/to/lesion.jpg",
    checkpoint_path="best_model_mixed.pth",
    use_medgemma=True,
)
print(report)
```

---

## Repository Structure

```
dermtriage/
  src/
    model.py           # MedSigLIPClassifier (7-class)
    dataset.py          # HAM10000Dataset, PADUFESDataset
    loss.py             # FocalLoss
    calibration.py      # TemperatureScaling, ECE
    gradcam.py          # Grad-CAM visual explainability
    explainer.py        # MedGemmaExplainer
    pipeline.py         # Full inference pipeline
  notebooks/
    01_train_baseline.ipynb    # Exp 1: HAM10000 only
    02_train_mixed.ipynb       # Exp 2: HAM + PAD mixed training
    03_demo_pipeline.ipynb     # Demo: classify + GradCAM + MedGemma
  docs/
    KAGGLE_WRITEUP.md          # Competition submission writeup
    RESULTS.md                 # Experiment results
  app.py                       # Gradio demo (HF Spaces)
  README.md
```

---

## Training Details

| Parameter | Exp 1 (Baseline) | Exp 2 (Mixed) |
|-----------|:-----------------:|:--------------:|
| Datasets | HAM10000 | HAM10000 + PAD-UFES-20 (3x oversample) |
| Epochs | 30 (early stop) | 30 (early stop) |
| Batch Size | 16 | 32 |
| Optimizer | AdamW (lr=1e-3) | AdamW (lr=1e-3) |
| Loss | Focal (gamma=2, 2x mel) | Focal (gamma=2, 2x mel) |
| Encoder | Frozen | Frozen |
| Trainable Params | 596K | 596K |
| Augmentation | HFlip, VFlip, Rot, ColorJitter | + GaussianBlur, stronger ColorJitter |

---

## Technical Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA-capable GPU (A100 recommended for training)
- HuggingFace account with access to MedSigLIP and MedGemma (gated models)

---

## Known Issues

**High referral rate.** The current best system (ensemble alpha=0.6) has a 63.4% referral rate on DDI. The root cause is OOD conditions: 53-56% of false positives come from conditions the model has never seen. Fitzpatrick17k binary gate (Exp 10c) is the current approach to address this.

**V-VI sensitivity below target.** Fitzpatrick V-VI sensitivity is 77-79% across models, below the 80% target. Feature fusion worsened this, suggesting DermLIP features dilute MedSigLIP's strength on darker skin tones.

**HAM10000 data leakage.** HAM10000 has 39.5% data leakage in naive random splits due to multiple images of the same lesion. All experiments from Notebook 09 onward use lesion-aware splits. Earlier experiments used naive splits and their validation metrics are inflated.

**Limited clinical diversity.** PAD-UFES-20 is a single clinical dataset from Brazil. DDI provides broader diversity but is evaluation-only (656 images). Real-world performance on images from other populations, phone cameras, or lighting conditions is not validated.

---

## References

1. Tschandl P, et al. "The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions." *Scientific Data*, 2018.
2. Pacheco AG, et al. "PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones." *Data in Brief*, 2020.
3. Daneshjou R, et al. "Disparities in dermatology AI performance on a diverse, curated clinical image set." *Science Advances*, 2022.
4. Ward A, et al. "Creating an Empirical Dermatology Dataset Through Crowdsourcing With Web Search Advertisements." *JAMA Network Open*, 2024.
5. Guo C, et al. "On Calibration of Modern Neural Networks." *ICML*, 2017.
6. Selvaraju RR, et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV*, 2017.

---

## Acknowledgments

- Google Health AI team for MedSigLIP and MedGemma models
- HAM10000, PAD-UFES-20, and DDI dataset authors
- HuggingFace for the transformers library

---

## Disclaimer

**This project is for research and education only.**

- This is NOT a medical device and must not be used for clinical decisions
- Results are from retrospective validation on research datasets
- Prospective clinical validation has not been performed
- Known limitations include high referral rate on OOD conditions and V-VI sensitivity below target
- Always involve qualified clinicians for any real-world evaluation

---

## License

MIT

---

*Last Updated: 2026-03-29*
