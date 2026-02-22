"""
End-to-end DermTriage inference pipeline.

Combines MedSigLIP classification, MC-Dropout uncertainty,
GradCAM explainability, and MedGemma clinical explanations
into a single pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms

from .model import MedSigLIPClassifier
from .explainer import MedGemmaExplainer, CLASS_INFO

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def preprocess_image(image_path, size=448):
    """Load and preprocess an image for MedSigLIP.

    Returns:
        (image_tensor [1, 3, H, W], pil_image)
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((size, size), Image.BILINEAR)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(image).unsqueeze(0), image


def classify_with_uncertainty(model, image_tensor, device=None, num_samples=10):
    """Classify with MC-Dropout uncertainty estimation.

    Returns:
        dict with class, confidence, uncertainty, risk_level, recommended_action, all_probs.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    image_tensor = image_tensor.to(device)

    # Main prediction
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=-1)
        confidence, pred_idx = probs.max(dim=-1)

    # MC Dropout for uncertainty
    def enable_dropout(m):
        if isinstance(m, nn.Dropout):
            m.train()

    model.classifier.apply(enable_dropout)
    mc_probs = []
    with torch.no_grad():
        for _ in range(num_samples):
            mc_probs.append(torch.softmax(model(image_tensor), dim=-1))

    mc_probs = torch.stack(mc_probs)
    mean_probs = mc_probs.mean(dim=0)

    entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1).item()
    normalized_entropy = entropy / np.log(len(CLASS_NAMES))

    pred_class = CLASS_NAMES[pred_idx.item()]
    info = CLASS_INFO[pred_class]

    return {
        "class": pred_class,
        "class_name": info["full_name"],
        "confidence": confidence.item(),
        "uncertainty": normalized_entropy,
        "risk_level": info["risk_level"],
        "recommended_action": info["action"],
        "all_probs": {CLASS_NAMES[i]: probs[0, i].item() for i in range(len(CLASS_NAMES))},
    }


def run_dermtriage_pipeline(image_path, checkpoint_path, device=None, use_medgemma=True):
    """Run the complete DermTriage pipeline on a single image.

    Args:
        image_path: Path to a dermoscopic or clinical skin lesion image.
        checkpoint_path: Path to a trained MedSigLIP ``.pth`` checkpoint.
        device: torch device (auto-detected if None).
        use_medgemma: Whether to generate a MedGemma explanation (needs more GPU RAM).

    Returns:
        (report_string, classification_result_dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load model
    model = MedSigLIPClassifier(num_classes=7).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "classifier_state_dict" in checkpoint:
        model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    # 2. Preprocess
    image_tensor, pil_image = preprocess_image(image_path)

    # 3. Classify
    result = classify_with_uncertainty(model, image_tensor, device=device)

    # 4. MedGemma explanation (optional)
    explanation = None
    if use_medgemma:
        explanation = _generate_explanation(pil_image, result)

    # 5. Format report
    report = format_clinical_report(result, explanation, image_path)
    return report, result


def _generate_explanation(image, classification_result):
    """Internal helper to generate MedGemma explanation."""
    explainer = MedGemmaExplainer()
    report = explainer.generate_explanation(
        image=image,
        predicted_class=classification_result["class"],
        confidence=classification_result["confidence"],
        uncertainty=classification_result.get("uncertainty"),
    )
    return report.get("ai_explanation", "")


def format_clinical_report(result, explanation, image_path):
    """Format classification results as a clinical report string."""
    lines = [
        "",
        "=" * 60,
        "DERMTRIAGE CLINICAL DECISION SUPPORT REPORT",
        "=" * 60,
        f"Image: {Path(image_path).name}",
        "",
        "-" * 60,
        "CLASSIFICATION RESULT",
        "-" * 60,
        f"Diagnosis: {result['class_name']}",
        f"Risk Level: {result['risk_level']}",
        f"Confidence: {result['confidence'] * 100:.1f}%",
        f"Uncertainty: {result['uncertainty']:.3f} ({'HIGH' if result['uncertainty'] > 0.3 else 'LOW'})",
        "",
        "Class Probabilities:",
    ]

    sorted_probs = sorted(result["all_probs"].items(), key=lambda x: x[1], reverse=True)
    for cls, prob in sorted_probs:
        name = CLASS_INFO[cls]["full_name"]
        marker = " <--" if cls == result["class"] else ""
        lines.append(f"  {name:42s}: {prob * 100:5.1f}%{marker}")

    lines += [
        "",
        "-" * 60,
        "RECOMMENDED ACTION",
        "-" * 60,
        result["recommended_action"],
    ]

    if explanation:
        lines += [
            "",
            "-" * 60,
            "AI CLINICAL ANALYSIS (MedGemma)",
            "-" * 60,
            explanation,
        ]

    lines += [
        "",
        "=" * 60,
        "DISCLAIMER: This report is for clinical decision support only.",
        "Final diagnosis requires expert dermatologic evaluation.",
        "=" * 60,
    ]
    return "\n".join(lines)
