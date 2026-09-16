# models/plantnet_model.py
import os
import json
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from .base_model import BasePlantModel
from core.config import WEIGHTS_DIR
from core.formatters import check_low_confidence_alternatives


class PlantNetModel(BasePlantModel):
    def __init__(self):
        super().__init__("PlantNet-300K")

        # Centralized paths
        self.model_dir = os.path.join(WEIGHTS_DIR, "plantnet300k")
        self.class_mapping_path = os.path.join(
            self.model_dir, "plantnet300k_class_mapping.json"
        )

        # Image pre-processing for standard PyTorch ResNet
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.classes = []

    def load(self, model_path=None):
        target_path = model_path if model_path and os.path.exists(model_path) else None

        # 1. Auto-detect any .pth, .pt, or .tar file in the directory if no explicit path is given
        if not target_path:
            possible_files = [
                f
                for f in os.listdir(self.model_dir)
                if f.endswith((".pth", ".tar", ".pt"))
            ]
            if possible_files:
                target_path = os.path.join(self.model_dir, possible_files[0])
            else:
                raise FileNotFoundError(
                    f"[{self.name}] No weights found! Please place your .tar or .pth file in {self.model_dir}"
                )

        print(f"[{self.name}] Loading weights from {target_path}...")

        # 2. Initialize ResNet18 architecture (Updated to match your specific downloaded file)
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 1081)

        # 3. Load the weights directly (PyTorch handles its own custom .tar files seamlessly)
        checkpoint = torch.load(target_path, map_location=torch.device("cpu"))

        # Checkpoint files are often saved as dictionaries.
        # This safely extracts just the weights based on common naming conventions.
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            elif "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif "model" in checkpoint:
                # This catches your specific resnet18_weights_best_acc.tar structure!
                self.model.load_state_dict(checkpoint["model"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        # 4. Load the two-step mapping correctly with safety checks
        idx_to_id_path = os.path.join(self.model_dir, "class_idx_to_species_id.json")
        id_to_name_path = os.path.join(
            self.model_dir, "plantnet300K_species_id_2_name.json"
        )

        if os.path.exists(idx_to_id_path) and os.path.exists(id_to_name_path):
            with open(idx_to_id_path, "r") as f:
                idx_to_species_id = json.load(f)
            with open(id_to_name_path, "r") as f:
                species_id_to_name = json.load(f)

            # Bridge them: Index (0-1080) -> Species ID -> Scientific Name
            self.classes = {}
            for idx, spec_id in idx_to_species_id.items():
                # Ensure we check both string and int keys just in case
                name = species_id_to_name.get(str(spec_id)) or species_id_to_name.get(
                    spec_id
                )
                self.classes[str(idx)] = name if name else f"Unknown_Species_{spec_id}"

            print(
                f"[{self.name}] Successfully loaded {len(self.classes)} mapped species names."
            )
        else:
            print(
                f"[{self.name}] Warning: Mapping JSON files not found in {self.model_dir}. Using fallback indices."
            )
            self.classes = {str(i): f"Species_Index_{i}" for i in range(1081)}

    def predict(self, image_path):
        """Processes the image and conditionally shows runner-ups > 50% confidence if top-1 is < 90%."""
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_idx = torch.max(probs, dim=0)

        top1_idx = top_idx.item()
        species = self.classes.get(str(top1_idx), f"Species_Index_{top1_idx}")
        confidence = top_prob.item() * 100

        # Invoke centralized low-confidence checker
        check_low_confidence_alternatives(
            self.name, probs, lambda idx: self.classes.get(str(idx), f"Index_{idx}")
        )

        return species, confidence
