"""
Wrapper for juppy44 Vision Transformer.
"""

# models/juppy_model.py

import os
from typing import List, Tuple

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

from .base_model import BasePlantModel
from core.config import WEIGHTS_DIR
from core.formatters import check_low_confidence_alternatives


class JuppyModel(BasePlantModel):
    """Wrapper for the juppy44 Plant Identification ViT-B model."""

    def __init__(self):
        super().__init__("Juppy44 ViT-B")
        self.model_id = "juppy44/plant-identification-2m-vit-b"

        # Centralized path: ./weights/juppy44
        self.model_dir = os.path.join(WEIGHTS_DIR, "juppy44")

    def load(self, model_path=None):
        """Load the model and image processor."""

        # 1. Check if user provided an explicit local directory via the UI.
        if model_path and os.path.exists(model_path):
            print(f"[{self.name}] Loading from explicit local path: " f"{model_path}")

            self.processor = AutoImageProcessor.from_pretrained(model_path)
            self.model = AutoModelForImageClassification.from_pretrained(model_path)

        else:
            # 2. Try loading from the centralized local folder first.
            try:
                print(
                    f"[{self.name}] Checking centralized folder: "
                    f"{self.model_dir}..."
                )

                self.processor = AutoImageProcessor.from_pretrained(
                    self.model_dir,
                    local_files_only=True,
                )

                self.model = AutoModelForImageClassification.from_pretrained(
                    self.model_dir,
                    local_files_only=True,
                )

                print(f"[{self.name}] Successfully loaded from local folder!")

            # 3. Fall back to downloading into the centralized folder.
            except Exception:
                print(
                    f"[{self.name}] Model not found locally. "
                    f"Downloading to {self.model_dir}..."
                )

                self.processor = AutoImageProcessor.from_pretrained(
                    self.model_id,
                    cache_dir=self.model_dir,
                )

                self.model = AutoModelForImageClassification.from_pretrained(
                    self.model_id,
                    cache_dir=self.model_dir,
                )

        self.model.eval()

    def predict(
        self,
        image_path: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Predict the most likely plant species.

        Args:
            image_path:
                Path to the input image.

            top_k:
                Number of ranked predictions to return.

        Returns:
            List of (species_name, confidence_percentage) tuples,
            ordered from highest to lowest confidence.
        """

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        # Make sure top_k cannot exceed the number of classes.
        top_k = min(top_k, probs.shape[0])

        top_probs, top_indices = torch.topk(
            probs,
            k=top_k,
        )

        predictions = []

        for probability, index in zip(top_probs, top_indices):
            species = self.model.config.id2label[index.item()]
            confidence = probability.item() * 100

            predictions.append(
                (
                    species,
                    confidence,
                )
            )

        # Keep the existing low-confidence diagnostic.
        check_low_confidence_alternatives(
            self.name,
            probs,
            lambda idx: self.model.config.id2label[idx],
        )

        return predictions
