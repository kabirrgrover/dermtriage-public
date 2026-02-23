# DermTriage

**AI-powered clinical decision support for skin lesion triage, built for the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge).**

Combines **MedSigLIP** (vision classification) with **MedGemma** (clinical explanation) to classify 7 skin lesion types, estimate uncertainty via MC-Dropout, visualize model attention with Grad-CAM, and generate natural language clinical assessments.

[![Open Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kabirrgrover/dermtriage-public/blob/master/notebooks/03_demo_pipeline.ipynb) [![Open in HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Kabirgrover/dermtriage)

---

## Why DermTriage?

Melanoma kills nearly 60,000 people worldwide each year (GLOBOCAN 2022), yet when caught early the five-year survival rate exceeds 99%. The bottleneck is access: most skin checks happen in primary care, where physicians use smartphones — not dermatoscopes — and may see fewer than a dozen melanomas in their career.

DermTriage addresses this by:
- **Bridging the domain gap** between dermoscopic training data and real-world smartphone images (+31.9% cross-domain accuracy via mixed training)
- **Prioritizing safety** — the system is tuned for high melanoma recall (catching cancers) over precision (fewer false alarms), the right trade-off for a screening tool
- **Building clinician trust** through Grad-CAM heatmaps (showing *where* the model looks), MC-Dropout uncertainty (knowing *when* to defer), and MedGemma explanations (explaining *why* in clinical language)

### Safety-First Design

In our demo, a basal cell carcinoma image was classified as melanoma (URGENT) rather than BCC (HIGH). This is a **feature, not a bug** — both are cancers requiring referral, and the system errs toward the more dangerous diagnosis. Combined with HIGH uncertainty flagging, this triggers the correct clinical action: urgent dermatology referral.

---

## Architecture

```
                        Input Image (448x448)
                               |
                               v
                 +-----------------------------+
                 |    MedSigLIP-448 Vision     |  Frozen encoder (~429M params)
                 |         Encoder             |
                 +-------------+---------------+
                               |
                 +-------------+-------------+
                 |             |             |
                 v             v             v
          +------------+ +----------+ +-----------+
          | Classifier | | Grad-CAM | | MC Dropout|
          |  Head      | | Heatmap  | |Uncertainty|
          | (596K)     | |          | |           |
          +-----+------+ +----+-----+ +-----+-----+
                |             |             |
                +-------------+-------------+
                              |
                              v
                 +-----------------------------+
                 |   MedGemma-4B (optional)    |
                 |  Clinical NL Explanation    |
                 +-------------+---------------+
                               |
                               v
                    Clinical Decision Report
```

**Classification Head:** `LayerNorm -> Dropout(0.3) -> Linear(embed_dim, 512) -> GELU -> Dropout(0.3) -> Linear(512, 7)`

**Classes:** Actinic Keratosis, Basal Cell Carcinoma, Benign Keratosis, Dermatofibroma, Melanoma, Melanocytic Nevus, Vascular Lesion

---

## Key Results

Two experiments evaluate single-domain vs. mixed-domain training:

| Model | Training Data | HAM10K Bal Acc | HAM10K Mel Recall | PAD-UFES Bal Acc | PAD-UFES Mel Recall |
|-------|---------------|:--------------:|:-----------------:|:----------------:|:-------------------:|
| M1 (Baseline) | HAM10000 | **78.9%** | **85.7%** | 49.9% | 54.5% |
| M2 (Mixed) | HAM10000 + PAD-UFES-20 | 69.6% | 77.6% | **81.8%** | **81.8%** |

**M2 is the recommended model** for primary care smartphone triage: it trades 9.3 points of in-domain accuracy for **31.9 points of cross-domain accuracy**.

### Fairness (PAD-UFES-20, Fitzpatrick skin types)

| Fitzpatrick | M1 Mel Recall | M2 Mel Recall | Delta |
|:-----------:|:-------------:|:-------------:|:-----:|
| I-II | 50.0% | 75.0% | +25.0 |
| III-IV | 66.7% | **100.0%** | +33.3 |

Mixed training improved melanoma recall across skin tones, with the **largest gains for darker skin types** — inverting the typical dermatology AI bias pattern.

See [docs/RESULTS.md](docs/RESULTS.md) for full per-class breakdowns and fairness analysis.

---

## HAI-DEF Model Usage

DermTriage uses **two** models from Google's Health AI Developer Foundations:

| Model | Role | How Used |
|-------|------|----------|
| **MedSigLIP-448** | Vision encoder | Frozen backbone for 7-class skin lesion classification. Trained on medical image-text pairs — provides domain-specific features that outperform general-purpose encoders. |
| **MedGemma-4B-IT** | Clinical explainer | Generates natural language clinical assessments from dermoscopic images, incorporating classification results, confidence, and uncertainty context. |

MedSigLIP's medical pretraining is critical — it provides features that generalize across dermoscopic and clinical images without fine-tuning the encoder, enabling training with only 596K parameters on consumer GPUs.

---

## Quick Start

### Run the demo (recommended)

1. Open the [demo notebook](https://colab.research.google.com/github/kabirrgrover/dermtriage-public/blob/master/notebooks/03_demo_pipeline.ipynb) in Colab
2. Add your secrets in Colab (Settings > Secrets): `HF_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`
3. Run all cells — the checkpoint downloads automatically from Google Drive

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

## Repo Structure

```
dermtriage/
├── app.py                         # Gradio live demo (HF Spaces)
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE                        # MIT
├── src/
│   ├── model.py                   # MedSigLIPClassifier
│   ├── dataset.py                 # HAM10000Dataset, PADUFESDataset
│   ├── loss.py                    # FocalLoss
│   ├── calibration.py             # TemperatureScaling, ECE
│   ├── gradcam.py                 # GradCAM visual explanations
│   ├── explainer.py               # MedGemmaExplainer
│   └── pipeline.py                # Full inference pipeline
├── notebooks/
│   ├── 01_train_baseline.ipynb    # Experiment 1: HAM10000 baseline
│   ├── 02_train_mixed.ipynb       # Experiment 2: Mixed training
│   └── 03_demo_pipeline.ipynb     # Demo: classify + GradCAM + MedGemma
└── docs/
    ├── KAGGLE_WRITEUP.md          # Competition submission writeup
    └── RESULTS.md                 # Detailed experiment results
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

## Datasets

- **[HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)** — 10,015 dermoscopic images, 7 classes. Tschandl et al. (2018).
- **[PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1)** — 2,298 clinical smartphone images with Fitzpatrick metadata. Pacheco et al. (2020).

---

## Limitations

- **Not a medical device.** For research and educational purposes only.
- **Retrospective validation only.** Real-world performance may differ.
- **Limited Fitzpatrick V-VI data.** Neither dataset has sufficient dark skin representation.
- **Confidence calibration** has not been formally validated on the mixed model.
- Final diagnosis always requires expert dermatologic evaluation.

---

## Acknowledgments

- Google Health AI for MedSigLIP and MedGemma model releases
- HAM10000 and PAD-UFES-20 dataset authors
- Built with PyTorch, HuggingFace Transformers, and scikit-learn

## License

MIT
