"""Focal Loss for class-imbalanced skin lesion classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) with optional per-class weights.

    Down-weights easy examples so the model focuses on hard cases.
    Particularly useful for imbalanced datasets where melanoma is rare
    but must not be missed.

    Args:
        alpha: Per-class weight tensor, or None for uniform.
        gamma: Focusing parameter (default 2.0). Higher = more focus on hard examples.
    """

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()
