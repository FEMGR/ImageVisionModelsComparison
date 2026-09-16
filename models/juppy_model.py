"""
Wrapper for juppy44 Vision Transformer
"""
#models/juppy_model.py

import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from .base_model import BasePlantModel
from core.config import WEIGHTS_DIR
from core.formatters import check_low_confidence_alternatives


class JuppyModel(BasePlantModel):
    def __init__(self):
        super().__init__("Juppy44 ViT-B")
        self.model_id = "juppy44/plant-identification-2m-vit-b"

        # Centralized path: ./weights/juppy44
        self.model_dir = os.path.join(WEIGHTS_DIR, "juppy44")

    def load(self, model_path=None):
        # 1. Check if user provided an explicit local directory via the UI
        if model_path and os.path.exists(model_path):
            print(f"[{self.name}] Loading from explicit local path: {model_path}")
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            self.model = AutoModelForImageClassification.from_pretrained(model_path)

        else:
            # 2. Try loading from our custom centralized folder first
            try:
                print(f"[{self.name}] Checking centralized folder: {self.model_dir}...")
                self.processor = AutoImageProcessor.from_pretrained(self.model_dir, local_files_only=True)
                self.model = AutoModelForImageClassification.from_pretrained(self.model_dir, local_files_only=True)
                print(f"[{self.name}] Successfully loaded from local folder!")

            # 3. Fall back to downloading directly into our centralized folder
            except Exception:
                print(f"[{self.name}] Model not found locally. Downloading to {self.model_dir}...")
                self.processor = AutoImageProcessor.from_pretrained(self.model_id, cache_dir=self.model_dir)
                self.model = AutoModelForImageClassification.from_pretrained(self.model_id, cache_dir=self.model_dir)

        self.model.eval()

    def predict(self, image_path):
        """Processes the image and uses the shared alternative-checking utility."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = logits.softmax(dim=-1)[0]
            top_prob, top_idx = torch.max(probs, dim=0)

        top1_idx = top_idx.item()
        label = self.model.config.id2label[top1_idx]
        confidence = top_prob.item() * 100

        # Invoke centralized low-confidence checker
        check_low_confidence_alternatives(
            self.name,
            probs,
            lambda idx: self.model.config.id2label[idx]
        )

        return label, confidence