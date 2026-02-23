"""
DermTriage — Interactive Skin Lesion Triage Demo

Gradio app for the MedGemma Impact Challenge.
Classifies skin lesions into 7 categories using MedSigLIP, visualizes
model attention with Grad-CAM, and estimates uncertainty via MC-Dropout.

Runs on CPU (HuggingFace Spaces free tier, 16GB RAM).
MedGemma clinical explanations require GPU — see the Colab notebook.
"""

import os
import sys
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import MedSigLIPClassifier
from src.gradcam import GradCAM
from src.pipeline import classify_with_uncertainty
from src.explainer import CLASS_INFO

# --- Configuration ---
CHECKPOINT_GDRIVE_ID = "1xY8c0569DWqYx7otq6jJCN4v1RYmdSpj"
CHECKPOINT_PATH = "best_model_mixed.pth"
EXAMPLES_DIR = Path("examples")
RISK_EMOJI = {"URGENT": "🔴", "HIGH": "🟠", "MODERATE": "🟡", "LOW": "🟢"}


# --- Model loading ---

def download_checkpoint():
    """Download model checkpoint from Google Drive if not already present."""
    if Path(CHECKPOINT_PATH).exists():
        return
    import gdown
    print("Downloading checkpoint from Google Drive...")
    gdown.download(id=CHECKPOINT_GDRIVE_ID, output=CHECKPOINT_PATH, quiet=False)


def load_model():
    """Load MedSigLIPClassifier with pretrained weights on CPU."""
    download_checkpoint()
    device = torch.device("cpu")
    model = MedSigLIPClassifier(num_classes=7).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if "classifier_state_dict" in ckpt:
        model.classifier.load_state_dict(ckpt["classifier_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device


# --- Preprocessing ---

def preprocess(image, size=448):
    """PIL Image -> normalized tensor for MedSigLIP."""
    image = image.convert("RGB").resize((size, size), Image.BILINEAR)
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return t(image).unsqueeze(0), image


# --- Result formatting ---

def format_results(result):
    """Format classification result dict as Markdown."""
    cls = result["class"]
    info = CLASS_INFO[cls]
    risk = info["risk_level"]
    emoji = RISK_EMOJI.get(risk, "")
    unc = result["uncertainty"]
    unc_label = "HIGH ⚠️" if unc > 0.3 else "LOW ✓"

    md = f"## {emoji} {info['full_name']}\n\n"
    md += f"**Risk Level:** {risk} {emoji}  \n"
    md += f"**Confidence:** {result['confidence'] * 100:.1f}%  \n"
    md += f"**Uncertainty:** {unc:.3f} ({unc_label})\n\n"
    md += "---\n\n### All Class Probabilities\n\n"

    sorted_probs = sorted(
        result["all_probs"].items(), key=lambda x: x[1], reverse=True
    )
    for c, p in sorted_probs:
        name = CLASS_INFO[c]["full_name"]
        bar_len = int(p * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        marker = " ◀" if c == cls else ""
        md += f"`{bar}` **{p * 100:5.1f}%** {name}{marker}  \n"

    md += f"\n---\n\n### Recommended Action\n\n{info['action']}\n\n"

    if unc > 0.3:
        md += (
            "> ⚠️ **High uncertainty detected.** "
            "Consider expert review regardless of classification.\n\n"
        )
    return md


# --- Main analysis function ---

def analyze_lesion(image):
    """Classify a skin lesion image with Grad-CAM and uncertainty."""
    if image is None:
        return None, "⚠️ Please upload a skin lesion image to begin analysis."

    tensor, resized = preprocess(image)
    tensor = tensor.to(device)

    # Classification with MC-Dropout uncertainty
    result = classify_with_uncertainty(model, tensor, device=device, num_samples=10)

    # Grad-CAM visualization
    gradcam = GradCAM(model)
    overlay, heatmap, class_idx = gradcam.generate_visualization(resized, tensor)

    # Remove hooks to prevent accumulation across requests
    gradcam.target_layer._forward_hooks.clear()
    gradcam.target_layer._backward_hooks.clear()

    return overlay, format_results(result)


# --- Startup ---
print("Loading MedSigLIP model (this may take a minute on first run)...")
model, device = load_model()
print(f"Model loaded on {device}")

# Collect example images (if any exist in examples/ directory)
examples = []
if EXAMPLES_DIR.exists():
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        examples.extend(EXAMPLES_DIR.glob(ext))
    examples = [[str(p)] for p in sorted(examples)]


# --- Gradio UI ---
with gr.Blocks(title="DermTriage", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🔬 DermTriage — AI Skin Lesion Triage\n\n"
        "Upload a skin lesion image to get an AI-powered risk assessment "
        "with visual explanations.\n\n"
        "**Pipeline:** MedSigLIP classification → Grad-CAM attention map → "
        "MC-Dropout uncertainty estimation"
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Skin Lesion Image")
            analyze_btn = gr.Button(
                "🔍 Analyze Lesion", variant="primary", size="lg"
            )
        with gr.Column():
            output_image = gr.Image(type="pil", label="Grad-CAM Attention Map")

    output_text = gr.Markdown(label="Analysis Results")

    analyze_btn.click(
        fn=analyze_lesion,
        inputs=input_image,
        outputs=[output_image, output_text],
    )

    if examples:
        gr.Examples(
            examples=examples,
            inputs=input_image,
            label="Click an example to try",
        )

    gr.Markdown(
        "---\n\n"
        "**ℹ️ About this demo:** Runs MedSigLIP classification + Grad-CAM + "
        "MC-Dropout uncertainty on CPU (~10-15s per image). "
        "MedGemma clinical explanations require GPU and are available in the "
        "[full demo notebook](https://colab.research.google.com/github/"
        "kabirrgrover/dermtriage-public/blob/master/notebooks/"
        "03_demo_pipeline.ipynb). "
        "GPU hosting (e.g., HF Spaces with T4) would enable the complete "
        "pipeline including MedGemma natural language explanations.\n\n"
        "**⚠️ Disclaimer:** This tool is for research and educational purposes "
        "only. It is not a medical device. Final diagnosis always requires "
        "expert dermatologic evaluation."
    )

demo.launch()
