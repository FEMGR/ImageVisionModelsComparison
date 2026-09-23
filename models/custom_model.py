"""
Wrapper for user-selected local Hugging Face image-classification models.
"""

# models/custom_model.py

import os
from typing import List, Tuple

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

from .base_model import BasePlantModel
from core.formatters import check_low_confidence_alternatives


class CustomModel(BasePlantModel):
    """Wrapper for a custom local Hugging Face image-classification model."""

    def __init__(
        self,
        name: str = "Custom Model",
        model_path: str = None,
    ):
        super().__init__(name)
        self.model_path = model_path
        self.processor = None
        self.model = None

    def load(self, model_path: str = None):
        """Load the custom model and image processor from a local directory."""

        load_path = model_path or self.model_path

        if not load_path:
            raise ValueError(f"[{self.name}] No custom model folder was provided.")

        if not os.path.isdir(load_path):
            raise FileNotFoundError(
                f"[{self.name}] Custom model folder was not found: " f"{load_path}"
            )

        print(f"[{self.name}] Loading custom model from " f"{load_path}...")

        self.processor = AutoImageProcessor.from_pretrained(
            load_path,
            local_files_only=True,
        )

        self.model = AutoModelForImageClassification.from_pretrained(
            load_path,
            local_files_only=True,
        )

        self.model.eval()

        print(f"[{self.name}] Successfully loaded.")
        print(f"[{self.name}] Number of classes: " f"{self.model.config.num_labels}")

    def predict(
        self,
        image_path: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Predict the most likely species for the input image."""

        if self.model is None or self.processor is None:
            raise RuntimeError(
                f"[{self.name}] Model has not been loaded. "
                "Call load() before predict()."
            )

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(
                logits,
                dim=-1,
            )[0]

        top_k = min(
            top_k,
            probs.shape[0],
        )

        top_probs, top_indices = torch.topk(
            probs,
            k=top_k,
        )

        predictions = []

        for probability, index in zip(
            top_probs,
            top_indices,
        ):
            class_index = index.item()
            species = self.model.config.id2label[class_index]
            confidence = probability.item() * 100

            predictions.append(
                (
                    species,
                    confidence,
                )
            )

        check_low_confidence_alternatives(
            self.name,
            probs,
            lambda idx: self.model.config.id2label[idx],
        )

        return predictions
