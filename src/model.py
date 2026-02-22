"""
MedSigLIP Classifier for 7-class skin lesion classification.

Uses the vision encoder from google/medsiglip-448 with a trainable
classification head. The encoder is frozen by default — only the
596K-parameter head is trained.
"""

import torch
import torch.nn as nn


class MedSigLIPClassifier(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.3, freeze_encoder=True):
        super().__init__()
        from transformers import AutoModel

        full_model = AutoModel.from_pretrained(
            "google/medsiglip-448",
            torch_dtype=torch.float32,
        )
        self.vision_model = full_model.vision_model
        self.embed_dim = full_model.config.vision_config.hidden_size
        del full_model

        if freeze_encoder:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.vision_model(pixel_values=pixel_values)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(features)
