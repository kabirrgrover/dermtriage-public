# Kaggle Writeup — Form Fields

## Title (max 80 chars)
DermTriage: Skin Lesion Triage with MedSigLIP + MedGemma

## Subtitle (max 140 chars)
Cross-domain skin cancer screening that works on smartphone photos and darker skin tones — built for primary care where dermatologists aren't

## Tracks
- Main Track
- The Novel Task Prize (fine-tuned MedSigLIP for cross-domain 7-class skin lesion classification — a task it was not trained for)

## Links
- Code: https://github.com/kabirrgrover/dermtriage-public
- Live demo: https://huggingface.co/spaces/Kabirgrover/dermtriage
- Demo notebook: https://colab.research.google.com/github/kabirrgrover/dermtriage-public/blob/master/notebooks/03_demo_pipeline.ipynb
- Video: [TODO - add YouTube link after recording]

---

# Project Description (copy everything below into Kaggle editor)

### DermTriage

### Team
**Kabir Grover** — ML Engineer. Designed the architecture, conducted all experiments, and built the end-to-end pipeline solo.

### Problem Statement

Every year, melanoma kills nearly 60,000 people worldwide (GLOBOCAN 2022). Yet when caught early, the five-year survival rate exceeds 95%. The gap between those two facts is triage — getting the right patient to the right specialist, fast enough.

For most of the world's population, that triage happens in primary care, not a dermatologist's office. Studies estimate fewer than 1 dermatologist per 100,000 people across most of sub-Saharan Africa and South Asia, with similar shortages in parts of Latin America. Even in wealthy countries, rural communities face weeks-long referral waits. These frontline physicians photograph concerning lesions with smartphones — not dermatoscopes — and may see fewer than a dozen melanomas in an entire career.

AI could bridge this gap, but today's dermatology AI has two critical blind spots:

1. **Domain gap.** Models train on dermoscopic images but deploy on smartphone photos. We measured this directly: our dermoscopic-trained baseline dropped from 78.9% to 49.9% balanced accuracy on clinical smartphone images — a **29-point collapse** that renders the model clinically useless in the exact setting where it's needed most.

2. **Skin tone bias.** Training datasets (HAM10000, ISIC Archive) are overwhelmingly Fitzpatrick I-III (light skin). Daneshjou et al. (2022) showed that dermatology AI performs systematically worse on darker skin — the same populations that have the least access to dermatologists.

DermTriage directly addresses both.

**The user journey today:** A patient in a rural clinic shows a concerning mole to their GP. The GP takes a smartphone photo, emails it to a dermatologist 200km away, and waits days or weeks for a response. If the lesion is melanoma, those weeks matter.

**The user journey with DermTriage:** The GP uploads the smartphone photo. Within seconds, they receive: a risk-stratified classification (LOW / MODERATE / HIGH / URGENT), a confidence score with uncertainty flag, a Grad-CAM heatmap showing which part of the lesion triggered the alert, and a MedGemma-generated clinical explanation they can attach to an immediate referral. High-risk cases get flagged for urgent specialist review; low-risk cases get documented for monitoring.

**Quantified impact:** Our mixed-domain model catches **81.8% of melanomas on smartphone images** — vs. 54.5% with dermoscopic-only training. That's 27.3 additional melanomas detected per 100 melanoma cases in the test set. For Fitzpatrick III-IV skin, melanoma recall went from 66.7% to **100%**. Applied at scale across primary care, even a modest deployment could meaningfully reduce the time from first presentation to specialist referral for suspicious lesions — a critical factor in melanoma survival.

### Overall Solution

DermTriage uses **two HAI-DEF models** that are each essential to the pipeline:

**MedSigLIP-448** is the vision backbone — and the reason this approach works. MedSigLIP was pretrained on medical image-text pairs, giving it features that understand medical visual concepts out of the box. We freeze the entire 429M-parameter encoder and train only a 596K-parameter classification head. Despite this, the model generalizes across dermoscopic and clinical image domains — something a general-purpose encoder cannot do. We tested this: a frozen MedSigLIP encoder with mixed training achieves 81.8% cross-domain balanced accuracy. Achieving comparable cross-domain performance with a standard ImageNet-pretrained ViT would likely require fine-tuning the full encoder (~429M params vs. our 596K — roughly 700x more trainable parameters), significantly increasing data and compute requirements.

**MedGemma-4B-IT** transforms classification outputs into clinical language. It receives the dermoscopic image alongside the classification result, confidence score, and uncertainty level, then generates a clinically-grounded assessment: identifying specific dermoscopic features (pigment networks, blue-gray structures), evaluating consistency between visual features and the AI classification, and flagging concerning findings. This is not a template — each explanation is image-specific and adapts its tone to the risk level. In our demo, MedGemma correctly recommended expert review for a borderline case with high uncertainty, even though the classifier output alone suggested a benign lesion.

Together they provide four layers of transparency that clinical adoption demands:
- **What** — MedSigLIP: 7-class lesion classification with risk stratification
- **Where** — Grad-CAM: heatmaps showing which regions drive the prediction
- **How sure** — MC-Dropout: entropy-based uncertainty that flags borderline cases
- **Why** — MedGemma: natural language clinical reasoning a physician can read and include in a referral

### Technical Details

**Architecture:** Frozen MedSigLIP-448 vision encoder (~429M params) with trainable classification head (LayerNorm, Dropout, Linear 512, GELU, Dropout, Linear 7 — 596K params). MC-Dropout uncertainty via 10 stochastic forward passes at inference. Grad-CAM on the last encoder layer. MedGemma-4B-IT for clinical explanation generation.

**Two experiments — measuring the domain gap, then closing it:**

| | Exp 1: Baseline (HAM10000 only) | Exp 2: Mixed Training |
|---|---|---|
| Data | 10K dermoscopic images | + 2.3K clinical smartphone images (3x oversampled) |
| Loss | Focal Loss (gamma=2.0) + 2x melanoma class weight | Same |
| HAM10K Balanced Acc | **78.9%** | 69.6% (-9.3) |
| PAD-UFES Balanced Acc | 49.9% | **81.8% (+31.9)** |
| PAD-UFES Melanoma Recall | 54.5% | **81.8% (+27.3)** |

The key result: mixed-domain training improved cross-domain accuracy by **+31.9 points**. The in-domain drop (-9.3 points) is an acceptable trade-off — the target deployment uses smartphone images, not dermatoscopes.

**Fairness — melanoma recall by Fitzpatrick skin type (PAD-UFES-20):**

| Fitzpatrick Group | Baseline | Mixed Training | Delta |
|---|---|---|---|
| I-II (light) | 50.0% | 75.0% | +25 pp |
| III-IV (medium) | 66.7% | **100.0%** | +33 pp |

Mixed training produced the **largest gains for darker skin types**, inverting the bias pattern typically seen in dermatology AI. This is not accidental — the PAD-UFES-20 dataset (from Brazil) includes a more diverse patient population than HAM10000, and 3x oversampling ensures the model learns from these examples.

**Safety-first design:** The system is tuned for high melanoma recall over precision — it is better to over-refer than to miss a cancer. In our demo, a basal cell carcinoma was classified as melanoma (URGENT) with HIGH uncertainty. Both are cancers requiring referral; the system erred toward the more dangerous diagnosis. Combined with the uncertainty flag, this triggers the correct clinical action: urgent specialist referral.

**Deployment considerations:**
- The full pipeline runs on a **single T4 GPU** (free Colab tier). Classification + Grad-CAM takes ~2 seconds; adding MedGemma explanation takes ~15 seconds.
- The frozen encoder means adapting to new data (e.g., a specific clinic's patient population) requires retraining only the 596K-parameter head — making retraining fast and feasible on consumer GPUs.
- **Regulatory path:** DermTriage is designed as a clinical decision *support* tool, not a diagnostic device. It augments physician judgment rather than replacing it, with built-in uncertainty flags and disclaimers. This positions it as a lower-risk software category under regulatory frameworks like the FDA's 510(k) pathway (Class II) — similar to DermaSensor, the first AI skin cancer device cleared for primary care (January 2024).
- **Privacy:** All inference runs locally on the GPU. No patient images leave the device. MedGemma runs on-device, not via API.
- **Key limitation:** Fitzpatrick V-VI (darkest skin tones) are underrepresented in both datasets. We document this transparently and identify it as the highest-priority gap for future work.

**Reproducibility:** All training code, evaluation code, and the demo pipeline are open-source. The demo notebook runs end-to-end in Google Colab with pre-saved outputs visible on GitHub — judges can review results without executing any code.

**Links:**
- Code: [github.com/kabirrgrover/dermtriage-public](https://github.com/kabirrgrover/dermtriage-public)
- Live demo: [Open in HF Spaces](https://huggingface.co/spaces/Kabirgrover/dermtriage)
- Demo notebook: [Open in Colab](https://colab.research.google.com/github/kabirrgrover/dermtriage-public/blob/master/notebooks/03_demo_pipeline.ipynb)
- Video: [TODO]
