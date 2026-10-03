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


class HuggingFaceLogitsWrapper(nn.Module):
    """
    Present Hugging Face image classifiers as plain-logit PyTorch modules.

    The generic fine-tuner calls models as ``model(images)`` and expects
    logits. Hugging Face image classifiers expect ``pixel_values=images``
    and return an object with a ``logits`` attribute.
    """

    def __init__(
        self,
        model: nn.Module,
        keep_batchnorm_eval: bool = False,
    ):
        super().__init__()
        self.huggingface_model = model
        self.keep_batchnorm_eval = keep_batchnorm_eval

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits for a batch of image tensors."""

        return self.huggingface_model(pixel_values=images).logits

    def train(self, mode: bool = True):
        result = super().train(mode)

        if mode and self.keep_batchnorm_eval:
            self._freeze_batchnorm_layers()

        return result

    def _freeze_batchnorm_layers(self) -> None:
        """Keep BatchNorm running statistics fixed during training."""

        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()


class HuggingFaceVisionAdapter(VisionModelAdapter):
    """Adapter that lets the generic fine-tuner train Hugging Face models."""

    def __init__(
        self,
        model: nn.Module,
        keep_batchnorm_eval: bool = False,
    ):
        self.huggingface_model = model

        super().__init__(
            HuggingFaceLogitsWrapper(
                model=model,
                keep_batchnorm_eval=keep_batchnorm_eval,
            )
        )

    def replace_classifier(self, num_classes: int) -> None:
        """
        Classifier replacement is architecture-specific.

        Scripts can still use this adapter after they have expanded or
        replaced the classifier with their own model-specific logic.
        """

        raise NotImplementedError(
            "Replace the classifier with architecture-specific logic "
            "before constructing HuggingFaceVisionAdapter."
        )

    def get_huggingface_model(self) -> nn.Module:
        """Return the underlying Hugging Face model."""

        return self.huggingface_model


class TorchVisionLogitsWrapper(nn.Module):
    """
    Wrap torchvision models while preserving their plain-logit forward pass.
    """

    def __init__(
        self,
        model: nn.Module,
        keep_batchnorm_eval: bool = False,
    ):
        super().__init__()
        self.torchvision_model = model
        self.keep_batchnorm_eval = keep_batchnorm_eval

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits for a batch of image tensors."""

        return self.torchvision_model(images)

    def train(self, mode: bool = True):
        result = super().train(mode)

        if mode and self.keep_batchnorm_eval:
            self._freeze_batchnorm_layers()

        return result

    def _freeze_batchnorm_layers(self) -> None:
        """Keep BatchNorm running statistics fixed during training."""

        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()


class TorchVisionModelAdapter(VisionModelAdapter):
    """Adapter for torchvision-style models that already return logits."""

    def __init__(
        self,
        model: nn.Module,
        keep_batchnorm_eval: bool = False,
    ):
        self.torchvision_model = model

        super().__init__(
            TorchVisionLogitsWrapper(
                model=model,
                keep_batchnorm_eval=keep_batchnorm_eval,
            )
        )

    def replace_classifier(self, num_classes: int) -> None:
        """
        Classifier replacement is architecture-specific.

        Scripts can use this adapter after replacing or expanding their
        classifier with model-specific logic.
        """

        raise NotImplementedError(
            "Replace the classifier with architecture-specific logic "
            "before constructing TorchVisionModelAdapter."
        )

    def get_torchvision_model(self) -> nn.Module:
        """Return the underlying torchvision model."""

        return self.torchvision_model
