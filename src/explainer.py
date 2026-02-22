"""
MedGemma-powered clinical explanation generator.

Uses MedGemma (google/medgemma-4b-it) to produce natural-language
clinical assessments for skin lesion images, enhanced with
classification context from MedSigLIP.

Requires a HuggingFace token with access to MedGemma.
Set the ``HF_TOKEN`` environment variable or pass it to
``huggingface_hub.login()`` before calling ``load_model()``.
"""

import os
import torch
from PIL import Image
from pathlib import Path

CLASS_INFO = {
    "akiec": {
        "full_name": "Actinic Keratosis / Intraepithelial Carcinoma",
        "risk_level": "MODERATE",
        "description": "precancerous scaly lesion caused by sun damage",
        "action": "Dermatology referral within 2-4 weeks for evaluation and possible treatment",
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "risk_level": "HIGH",
        "description": "most common form of skin cancer, typically slow-growing",
        "action": "Dermatology referral within 2 weeks for biopsy and treatment planning",
    },
    "bkl": {
        "full_name": "Benign Keratosis",
        "risk_level": "LOW",
        "description": "non-cancerous growth including seborrheic keratosis",
        "action": "Routine monitoring; removal only if symptomatic or cosmetically desired",
    },
    "df": {
        "full_name": "Dermatofibroma",
        "risk_level": "LOW",
        "description": "benign fibrous skin nodule",
        "action": "No treatment required; reassure patient",
    },
    "mel": {
        "full_name": "Melanoma",
        "risk_level": "URGENT",
        "description": "potentially deadly form of skin cancer requiring immediate attention",
        "action": "URGENT dermatology referral within 48 hours; do not delay",
    },
    "nv": {
        "full_name": "Melanocytic Nevus",
        "risk_level": "LOW",
        "description": "common benign mole",
        "action": "Routine monitoring; educate patient on ABCDE warning signs",
    },
    "vasc": {
        "full_name": "Vascular Lesion",
        "risk_level": "LOW",
        "description": "benign blood vessel abnormality such as angioma",
        "action": "No treatment required unless symptomatic",
    },
}


class MedGemmaExplainer:
    """Lazy-loaded MedGemma explainer for clinical skin lesion analysis."""

    def __init__(self):
        self.model = None
        self.processor = None

    def load_model(self):
        """Load MedGemma (idempotent)."""
        if self.model is not None:
            return

        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.model = AutoModelForImageTextToText.from_pretrained(
            "google/medgemma-4b-it",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")

    def generate_explanation(self, image, predicted_class, confidence, uncertainty=None):
        """Generate a clinical explanation for a classified skin lesion.

        Args:
            image: PIL Image of the lesion.
            predicted_class: One of the CLASS_INFO keys (e.g. ``"mel"``).
            confidence: Model confidence in [0, 1].
            uncertainty: Optional uncertainty score.

        Returns:
            dict with classification details, AI explanation, and recommendation.
        """
        self.load_model()

        info = CLASS_INFO.get(predicted_class, CLASS_INFO["nv"])

        prompt = (
            "You are a dermatology AI assistant helping primary care physicians triage skin lesions.\n\n"
            "Analyze this dermoscopic image and provide a clinical assessment.\n\n"
            f"The AI classification system has identified this lesion as: {info['full_name']}\n"
            f"Classification confidence: {confidence * 100:.1f}%\n"
        )
        if uncertainty is not None:
            level = "HIGH - consider expert review" if uncertainty > 0.3 else "LOW"
            prompt += f"Uncertainty level: {level}\n"
        prompt += (
            "\nPlease provide:\n"
            "1. A brief description of the visible dermoscopic features (2-3 sentences)\n"
            "2. Whether the AI classification appears consistent with the visual features\n"
            "3. Any additional observations relevant to clinical decision-making\n\n"
            "Keep your response concise and clinically focused."
        )

        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=300, do_sample=False)
            generation = generation[0][input_len:]

        explanation = self.processor.decode(generation, skip_special_tokens=True)

        report = {
            "classification": info["full_name"],
            "class_code": predicted_class,
            "confidence": confidence,
            "risk_level": info["risk_level"],
            "ai_explanation": explanation,
            "recommended_action": info["action"],
            "clinical_description": info["description"],
        }

        if uncertainty is not None:
            report["uncertainty"] = uncertainty
            report["uncertainty_note"] = (
                "HIGH uncertainty - recommend expert review regardless of classification"
                if uncertainty > 0.3
                else "Uncertainty within acceptable range"
            )
        return report

    def format_report(self, report):
        """Format a report dict as a readable clinical summary string."""
        lines = [
            "=" * 60,
            "DERMTRIAGE CLINICAL DECISION SUPPORT REPORT",
            "=" * 60,
            "",
            f"CLASSIFICATION: {report['classification']}",
            f"RISK LEVEL: {report['risk_level']}",
            f"CONFIDENCE: {report['confidence'] * 100:.1f}%",
        ]
        if "uncertainty" in report:
            lines.append(f"UNCERTAINTY: {report['uncertainty']:.2f} - {report['uncertainty_note']}")
        lines += [
            "",
            "-" * 60,
            "AI ANALYSIS:",
            "-" * 60,
            report["ai_explanation"],
            "",
            "-" * 60,
            "RECOMMENDED ACTION:",
            "-" * 60,
            report["recommended_action"],
            "",
            "=" * 60,
            "This report is for clinical decision support only.",
            "Final diagnosis requires expert dermatologic evaluation.",
            "=" * 60,
        ]
        return "\n".join(lines)


def generate_referral_packet(image_path, classification_result):
    """Generate a complete referral packet from an image path and classification result.

    Args:
        image_path: Path to skin lesion image.
        classification_result: dict with ``class``, ``confidence``, and optionally ``uncertainty``.

    Returns:
        Formatted clinical report string.
    """
    image = Image.open(image_path).convert("RGB")
    explainer = MedGemmaExplainer()
    report = explainer.generate_explanation(
        image=image,
        predicted_class=classification_result["class"],
        confidence=classification_result["confidence"],
        uncertainty=classification_result.get("uncertainty"),
    )
    return explainer.format_report(report)
