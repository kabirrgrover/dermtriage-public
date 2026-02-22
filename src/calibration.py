"""
Temperature scaling for calibrated confidence scores.

Ensures that when the model says "80% confident", it is actually
correct ~80% of the time.  Critical for clinical trust.

Reference: Guo et al., "On Calibration of Modern Neural Networks" (2017)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


class TemperatureScaling(nn.Module):
    """Learn a single temperature parameter on a validation set."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def calibrate(self, model, val_loader, device, max_iter=50):
        """Optimize temperature via LBFGS on the validation NLL.

        Returns:
            Optimal temperature (float).
        """
        model.eval()
        self.to(device)

        all_logits, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Collecting logits"):
                logits = model(images.to(device))
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        all_logits = torch.cat(all_logits).to(device)
        all_labels = torch.cat(all_labels).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(all_logits), all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return self.temperature.item()

    def get_calibrated_probs(self, logits):
        return torch.softmax(self.forward(logits), dim=-1)


def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error. Lower is better; < 0.05 is well-calibrated."""
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_acc = accuracies[in_bin].float().mean()
            bin_conf = confidences[in_bin].mean()
            ece += (in_bin.sum().float() / len(labels)) * torch.abs(bin_acc - bin_conf)
    return ece.item()


def compute_reliability_diagram(probs, labels, n_bins=10):
    """Return data for a reliability diagram."""
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_confidences, bin_accuracies, bin_counts = [], [], []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_confidences.append(confidences[in_bin].mean().item())
            bin_accuracies.append(accuracies[in_bin].mean().item())
            bin_counts.append(in_bin.sum().item())
        else:
            bin_confidences.append((bin_boundaries[i] + bin_boundaries[i + 1]).item() / 2)
            bin_accuracies.append(0)
            bin_counts.append(0)

    return {
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
        "ece": compute_ece(probs, labels, n_bins),
    }
