# DermTriage

Clinical decision support for skin lesion triage using Google's medical foundation models.

Combines **MedSigLIP** (vision classification) with **MedGemma** (clinical explanation) to classify 7 skin lesion types, estimate uncertainty, and generate natural language clinical assessments.

[![Open Demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kabirrgrover/dermtriage-public/blob/master/notebooks/03_demo_pipeline.ipynb)

---

## Architecture

```
                        Input Image (448x448)
                               │
                               ▼
                 ┌───────────────────────────┐
                 │    MedSigLIP-448 Vision    │  Frozen encoder (~429M params)
                 │         Encoder            │
                 └─────────────┬─────────────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                 ▼             ▼             ▼
          ┌────────────┐ ┌──────────┐ ┌───────────┐
          │ Classifier │ │ Grad-CAM │ │ MC Dropout│
          │  Head      │ │ Heatmap  │ │Uncertainty│
          │ (596K)     │ │          │ │           │
          └─────┬──────┘ └────┬─────┘ └─────┬─────┘
                │             │             │
                └─────────────┼─────────────┘
                              │
                              ▼
                 ┌───────────────────────────┐
                 │   MedGemma-4B (optional)   │
                 │  Clinical NL Explanation   │
                 └─────────────┬─────────────┘
                               │
                               ▼
                    Clinical Decision Report
```

**Classification Head:**
`LayerNorm → Dropout(0.3) → Linear(embed_dim, 512) → GELU → Dropout(0.3) → Linear(512, 7)`

**Classes:** Actinic Keratosis, Basal Cell Carcinoma, Benign Keratosis, Dermatofibroma, Melanoma, Melanocytic Nevus, Vascular Lesion

---

## Key Results

Two experiments evaluate single-domain vs. mixed-domain training:

| Model | Training Data | HAM10K Bal Acc | HAM10K Mel Recall | PAD-UFES Bal Acc | PAD-UFES Mel Recall |
|-------|---------------|:--------------:|:-----------------:|:----------------:|:-------------------:|
| M1 (Baseline) | HAM10000 | **78.9%** | **85.7%** | 49.9% | 54.5% |
| M2 (Mixed) | HAM10000 + PAD-UFES-20 | 69.6% | 77.6% | **81.8%** | **81.8%** |

**M2 is the recommended model** for primary care smartphone triage: it trades 9.3 points of in-domain accuracy for 31.9 points of cross-domain accuracy.

### Fairness (PAD-UFES-20, Fitzpatrick skin types)

| Fitzpatrick | M1 Mel Recall | M2 Mel Recall |
|:-----------:|:-------------:|:-------------:|
| I-II | 50.0% | 75.0% |
| III-IV | 66.7% | **100.0%** |

Mixed training improved melanoma recall across skin tones, with the largest gains for darker skin types.

See [docs/RESULTS.md](docs/RESULTS.md) for full per-class breakdowns.

---

## Quick Start

### Run the demo (recommended)

1. Open the [demo notebook](https://colab.research.google.com/github/kabirrgrover/dermtriage-public/blob/master/notebooks/03_demo_pipeline.ipynb) in Colab
2. Add your HuggingFace token as a Colab secret (`HF_TOKEN`)
3. Upload the trained checkpoint (`best_model.pth`)
4. Run all cells

### Local setup

```bash
git clone https://github.com/kabirrgrover/dermtriage-public.git
cd dermtriage
pip install -r requirements.txt
```

```python
from src.pipeline import run_dermtriage_pipeline

report, result = run_dermtriage_pipeline(
    image_path="path/to/lesion.jpg",
    checkpoint_path="best_model.pth",
    use_medgemma=True,
)
print(report)
```

---

## Repo Structure

```
dermtriage/
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
