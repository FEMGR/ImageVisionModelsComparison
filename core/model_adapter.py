"""
Model adapter interface for vision fine-tuning.

The fine-tuning engine communicates with models through this interface
instead of depending on a specific architecture such as ResNet or ViT.
"""

# core/model_adapter.py


from abc import ABC, abstractmethod

import torch
from torch import nn


class VisionModelAdapter(ABC):
    """Interface for adapting pretrained vision models."""

    def __init__(self, model: nn.Module):
        self.model = model

    @abstractmethod
    def replace_classifier(self, num_classes: int) -> None:
        """
        Replace or modify the classification head.

        Args:
            num_classes: Number of classes in the new dataset.
        """
        raise NotImplementedError

    def freeze_backbone(self) -> None:
        """Freeze all model parameters."""
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def unfreeze_all(self) -> None:
        """Make all model parameters trainable."""
        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def trainable_parameters(self):
        """Return parameters that should be optimized."""
        return (
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )

    def to(self, device: torch.device) -> nn.Module:
        """Move model to the requested device."""
        return self.model.to(device)

    def get_model(self) -> nn.Module:
        """Return the wrapped model."""
        return self.model
