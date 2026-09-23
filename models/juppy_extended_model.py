"""
Wrapper class for the fine-tuned Juppy44 plant-identification model.
"""

# models/juppy_extended_model.py

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


class JuppyExtendedModel(BasePlantModel):
    """
    Wrapper for the fine-tuned Juppy44 model.

    This model is derived from the original Juppy44 pretrained
    model and is intended to provide additional recognition
    capability for locally relevant plant species.
    """

    def __init__(self):
        super().__init__("Juppy44 ViT-B + Local Plants")

        # Directory containing the fine-tuned Hugging Face model.
        self.model_dir = os.path.join(
            WEIGHTS_DIR,
            "juppy44_extended",
        )

    def load(self, model_path=None):
        """
        Load the fine-tuned model and image processor.

        Args:
            model_path:
                Optional explicit local model directory.
                If omitted, the centralized juppy44_extended
                directory is used.
        """

        # -----------------------------------------------------
        # 1. Determine model location
        # -----------------------------------------------------

        load_path = model_path or self.model_dir

        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"[{self.name}] Fine-tuned model was not found at: " f"{load_path}"
            )

        print(f"[{self.name}] Loading fine-tuned model from " f"{load_path}...")

        # -----------------------------------------------------
        # 2. Load processor
        # -----------------------------------------------------

        self.processor = AutoImageProcessor.from_pretrained(
            load_path,
            local_files_only=True,
        )

        # -----------------------------------------------------
        # 3. Load fine-tuned model
        # -----------------------------------------------------

        self.model = AutoModelForImageClassification.from_pretrained(
            load_path,
            local_files_only=True,
        )

        # -----------------------------------------------------
        # 4. Evaluation mode
        # -----------------------------------------------------

        self.model.eval()

        print(f"[{self.name}] Successfully loaded.")

        print(f"[{self.name}] Number of classes: " f"{self.model.config.num_labels}")

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
            List of (species_name, confidence_percentage)
            tuples ordered from highest to lowest confidence.
        """

        if self.model is None or self.processor is None:
            raise RuntimeError(
                f"[{self.name}] Model has not been loaded. "
                "Call load() before predict()."
            )

        # -----------------------------------------------------
        # 1. Load image
        # -----------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        # -----------------------------------------------------
        # 2. Prepare model input
        # -----------------------------------------------------

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        # -----------------------------------------------------
        # 3. Run inference
        # -----------------------------------------------------

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(
                logits,
                dim=-1,
            )[0]

        # -----------------------------------------------------
        # 4. Limit top-k to available classes
        # -----------------------------------------------------

        top_k = min(
            top_k,
            probs.shape[0],
        )

        top_probs, top_indices = torch.topk(
            probs,
            k=top_k,
        )

        # -----------------------------------------------------
        # 5. Convert predictions to species names
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # 6. Existing low-confidence diagnostic
        # -----------------------------------------------------

        check_low_confidence_alternatives(
            self.name,
            probs,
            lambda idx: self.model.config.id2label[idx],
        )

        return predictions
