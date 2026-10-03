"""
Wrapper for the extended PlantNet-300K model.
"""

# models/plantnet_extended_model.py

import json
import os
from typing import List, Tuple

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from .base_model import BasePlantModel
from core.config import WEIGHTS_DIR
from core.formatters import check_low_confidence_alternatives


class PlantNetExtendedModel(BasePlantModel):
    """
    Wrapper for the PlantNet-300K model extended with local plant classes.
    """

    def __init__(self):
        super().__init__("PlantNet-300K + Local Plants")

        self.model_dir = os.path.join(
            WEIGHTS_DIR,
            "plantnet300k_extended",
        )

        self.model_path = os.path.join(
            self.model_dir,
            "plantnet_resnet18_extended.pt",
        )

        self.id_to_label_path = os.path.join(
            self.model_dir,
            "extended_id2label.json",
        )

        self.label_to_id_path = os.path.join(
            self.model_dir,
            "extended_label2id.json",
        )

        self.local_classes_path = os.path.join(
            self.model_dir,
            "local_classes.json",
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ]
        )

        self.classes = {}

    def load(self, model_path=None):
        """
        Load the extended PlantNet model and class mappings.
        """

        target_path = (
            model_path if model_path and os.path.exists(model_path) else self.model_path
        )

        if not os.path.exists(target_path):
            raise FileNotFoundError(
                f"[{self.name}] Extended model weights " f"not found: {target_path}"
            )

        print(f"[{self.name}] Loading weights from " f"{target_path}...")

        # ---------------------------------------------------------
        # 1. Load checkpoint
        # ---------------------------------------------------------

        checkpoint = torch.load(
            target_path,
            map_location=torch.device("cpu"),
        )

        if not isinstance(checkpoint, dict):
            raise ValueError(
                f"[{self.name}] Invalid extended model " "checkpoint format."
            )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                f"[{self.name}] Extended checkpoint does not "
                "contain 'model_state_dict'."
            )

        state_dict = checkpoint["model_state_dict"]

        # ---------------------------------------------------------
        # 2. Determine number of classes
        # ---------------------------------------------------------

        if "num_classes" in checkpoint:
            num_classes = int(checkpoint["num_classes"])

        else:
            classifier_weight = state_dict.get("fc.weight")

            if classifier_weight is None:
                raise KeyError(
                    f"[{self.name}] Could not determine "
                    "the number of classes from checkpoint."
                )

            num_classes = classifier_weight.shape[0]

        print(f"[{self.name}] Number of classes: " f"{num_classes}")

        # ---------------------------------------------------------
        # 3. Create matching ResNet-18 architecture
        # ---------------------------------------------------------

        self.model = models.resnet18(weights=None)

        num_ftrs = self.model.fc.in_features

        self.model.fc = torch.nn.Linear(
            num_ftrs,
            num_classes,
        )

        # ---------------------------------------------------------
        # 4. Load extended weights
        # ---------------------------------------------------------

        self.model.load_state_dict(state_dict)

        self.model.eval()

        # ---------------------------------------------------------
        # 5. Load class mappings
        # ---------------------------------------------------------

        if os.path.exists(self.id_to_label_path):

            with open(
                self.id_to_label_path,
                "r",
                encoding="utf-8",
            ) as file:

                self.classes = json.load(file)

            # JSON object keys are strings.
            self.classes = {str(index): label for index, label in self.classes.items()}

        else:

            self.classes = {
                str(index): f"Class_{index}" for index in range(num_classes)
            }

            print(f"[{self.name}] Warning: " "extended_id2label.json not found.")

        # ---------------------------------------------------------
        # 6. Load local class information if available
        # ---------------------------------------------------------

        self.local_classes = []

        if os.path.exists(self.local_classes_path):

            with open(
                self.local_classes_path,
                "r",
                encoding="utf-8",
            ) as file:

                self.local_classes = json.load(file)

        print(f"[{self.name}] Successfully loaded " f"{len(self.classes)} classes.")

        print(f"[{self.name}] Local classes: " f"{len(self.local_classes)}")

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
                Number of ranked predictions.

        Returns:
            List of
            (species_name, confidence_percentage)
            tuples.
        """

        if self.model is None:
            raise RuntimeError(f"[{self.name}] Model has not been loaded.")

        image = Image.open(image_path).convert("RGB")

        input_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():

            output = self.model(input_tensor)

            probs = torch.softmax(
                output[0],
                dim=0,
            )

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

            idx = index.item()

            species = self.classes.get(
                str(idx),
                f"Class_{idx}",
            )

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
            lambda idx: self.classes.get(
                str(idx),
                f"Class_{idx}",
            ),
        )

        return predictions
