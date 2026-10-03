"""
Fine-tune ResNet-18 with additional local plant classes.

This script extends the original ResNet-18 classification head while
preserving the pretrained classifier weights for the original classes.

Only the newly added local-plant classifier rows are trained.
The original ResNet-18 backbone and original classifier rows remain frozen.

The original model is never modified.

Input:
    datasets/local_plants/
        train/
            class_a/
            class_b/
            ...
        validation/
            class_a/
            class_b/
            ...

Output:
    weights/plantnet300k_extended/
        plantnet_resnet18_extended.pt
        class_idx_to_species_id.json
        plantnet300K_species_id_2_name.json
        local_classes.json
"""

# fine_tune/resnet18_fineTuner.py

import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from core.analytics import (
    TrainingHistory,
    plot_roc_auc_curve,
    plot_training_accuracy,
    plot_training_loss,
    print_roc_auc,
)
from core.fine_tuner import FineTuner
from core.fine_tuning import FineTuneConfig
from core.formatters import (
    export_evaluation_results,
    export_evaluation_summary,
    export_training_history,
)
from core.model_adapter import TorchVisionModelAdapter

# =============================================================
# Configuration
# =============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

BASE_MODEL_DIR = ROOT_DIR / "weights" / "plantnet300k"

EXTENDED_MODEL_DIR = ROOT_DIR / "weights" / "plantnet300k_extended"

RESULTS_DIR = ROOT_DIR / "results" / "fine_tune" / "plantnet300k_extended"

DATASET_DIR = ROOT_DIR / "datasets" / "local_plants"

TRAIN_DIR = DATASET_DIR / "train"

VALIDATION_DIR = DATASET_DIR / "validation"

BATCH_SIZE = 16

EPOCHS = 30

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 0.0

RANDOM_SEED = 42


# =============================================================
# Reproducibility
# =============================================================


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================
# Device
# =============================================================


def get_device() -> torch.device:
    """Select CUDA when available, otherwise CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# =============================================================
# Dataset
# =============================================================


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


class LocalPlantDataset(Dataset):
    """
    Image dataset for local plant classification.
    """

    def __init__(
        self,
        root_dir: Path,
        transform,
        label2id: Dict[str, int],
    ):
        self.root_dir = Path(root_dir)

        self.transform = transform

        self.label2id = label2id

        self.samples: List[Tuple[Path, int]] = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        self._load_samples()

        if not self.samples:
            raise ValueError(f"No images found in {self.root_dir}")

    def _load_samples(self) -> None:
        """Discover images and assign class labels."""

        for class_dir in sorted(self.root_dir.iterdir()):

            if not class_dir.is_dir():
                continue

            class_name = class_dir.name

            if class_name not in self.label2id:
                raise ValueError(
                    f"Unknown class '{class_name}' found in {self.root_dir}"
                )

            label_id = self.label2id[class_name]

            for image_path in sorted(class_dir.rglob("*")):

                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    self.samples.append(
                        (
                            image_path,
                            label_id,
                        )
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label_id = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        pixel_values = self.transform(image)

        return (
            pixel_values,
            torch.tensor(
                label_id,
                dtype=torch.long,
            ),
        )


# =============================================================
# Model inspection
# =============================================================


def get_plantnet_transform():
    """Return the PlantNet ResNet-18 preprocessing transform."""

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def find_plantnet_weights() -> Path:
    """Find the local PlantNet checkpoint file."""

    preferred_files = [
        BASE_MODEL_DIR / "plantnet_resnet18.pth",
        BASE_MODEL_DIR / "resnet18_weights_best_acc.tar",
    ]

    for path in preferred_files:
        if path.exists():
            return path

    possible_files = sorted(
        path
        for path in BASE_MODEL_DIR.iterdir()
        if path.suffix.lower() in {".pth", ".pt", ".tar"}
    )

    if not possible_files:
        raise FileNotFoundError(
            "No PlantNet ResNet-18 weights found in " f"{BASE_MODEL_DIR}"
        )

    return possible_files[0]


def load_state_dict_from_checkpoint(checkpoint_path: Path):
    """Load a state dict from common PyTorch checkpoint shapes."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
    )

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint:
                return checkpoint[key]

    return checkpoint


def load_plantnet_model() -> nn.Module:
    """Load the local PlantNet-300K ResNet-18 checkpoint."""

    checkpoint_path = find_plantnet_weights()

    print(f"Loading PlantNet-300K weights from {checkpoint_path}...")

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        1081,
    )

    model.load_state_dict(load_state_dict_from_checkpoint(checkpoint_path))

    return model


def get_existing_labels() -> Tuple[Dict[int, str], Dict[str, int]]:
    """Read the original ResNet-18 class mappings."""

    idx_to_id_path = BASE_MODEL_DIR / "class_idx_to_species_id.json"

    id_to_name_path = BASE_MODEL_DIR / "plantnet300K_species_id_2_name.json"

    if not idx_to_id_path.exists() or not id_to_name_path.exists():
        id2label = {index: f"Species_Index_{index}" for index in range(1081)}

        label2id = {label: index for index, label in id2label.items()}

        return id2label, label2id

    with open(
        idx_to_id_path,
        "r",
        encoding="utf-8",
    ) as file:
        idx_to_species_id = json.load(file)

    with open(
        id_to_name_path,
        "r",
        encoding="utf-8",
    ) as file:
        species_id_to_name = json.load(file)

    id2label = {}

    for index, species_id in idx_to_species_id.items():
        label = species_id_to_name.get(
            str(species_id),
            f"Unknown_Species_{species_id}",
        )

        id2label[int(index)] = label

    label2id = {label: index for index, label in id2label.items()}

    return id2label, label2id


# =============================================================
# Local classes
# =============================================================


def discover_local_classes() -> List[str]:
    """Discover local plant classes from the training directory."""

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training dataset not found: {TRAIN_DIR}")

    classes = sorted(
        directory.name for directory in TRAIN_DIR.iterdir() if directory.is_dir()
    )

    if not classes:
        raise ValueError(f"No plant classes found in {TRAIN_DIR}")

    return classes


def validate_dataset_classes(local_classes: List[str]) -> None:
    """Make sure validation contains the same local classes as training."""

    if not VALIDATION_DIR.exists():
        raise FileNotFoundError(f"Validation directory not found: {VALIDATION_DIR}")

    validation_classes = sorted(
        directory.name for directory in VALIDATION_DIR.iterdir() if directory.is_dir()
    )

    missing_classes = sorted(set(local_classes) - set(validation_classes))

    if missing_classes:
        raise ValueError(
            "The following training classes are missing from validation:\n"
            f"{missing_classes}"
        )


# =============================================================
# Classifier expansion
# =============================================================


def get_classifier_layer(model) -> Tuple[str, nn.Linear]:
    """Locate the final linear classification layer for torchvision ResNet."""

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return "fc", model.fc

    raise TypeError("Unable to locate model.fc linear layer in ResNet model.")


def expand_classifier(
    model,
    local_classes: List[str],
) -> Tuple[Dict[int, str], Dict[str, int], int]:
    """Expand the ResNet-18 classifier with local plant classes."""

    old_id2label, old_label2id = get_existing_labels()

    old_num_classes = len(old_id2label)

    duplicate_classes = [
        class_name for class_name in local_classes if class_name in old_label2id
    ]

    if duplicate_classes:
        raise ValueError(
            "The following local classes already exist in model: "
            f"{duplicate_classes}"
        )

    _, old_classifier = get_classifier_layer(model)

    old_num_classes_from_layer = old_classifier.out_features

    if old_num_classes_from_layer != old_num_classes:
        raise ValueError(
            "Model configuration and classifier disagree about class count: "
            f"config={old_num_classes}, classifier={old_num_classes_from_layer}"
        )

    new_id2label = dict(old_id2label)
    new_label2id = dict(old_label2id)

    for offset, class_name in enumerate(local_classes):
        new_id = old_num_classes + offset
        new_id2label[new_id] = class_name
        new_label2id[class_name] = new_id

    new_num_classes = len(new_id2label)

    print(f"Original classes: {old_num_classes}")
    print(f"Local classes:    {len(local_classes)}")
    print(f"Extended classes: {new_num_classes}")

    new_classifier = nn.Linear(
        old_classifier.in_features,
        new_num_classes,
        bias=old_classifier.bias is not None,
    )

    with torch.no_grad():
        new_classifier.weight[:old_num_classes].copy_(old_classifier.weight)
        if old_classifier.bias is not None:
            new_classifier.bias[:old_num_classes].copy_(old_classifier.bias)

    model.fc = new_classifier

    return (
        new_id2label,
        new_label2id,
        old_num_classes,
    )


# =============================================================
# Freeze model & BatchNorm
# =============================================================


def freeze_batchnorm_layers(model: nn.Module) -> None:
    """Ensure all BatchNorm layers remain in eval mode during training."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


def freeze_model_except_new_classes(
    model,
    old_num_classes: int,
) -> None:
    """Freeze backbone and original classifier rows."""

    for parameter in model.parameters():
        parameter.requires_grad = False

    _, linear_layer = get_classifier_layer(model)

    linear_layer.weight.requires_grad = True

    if linear_layer.bias is not None:
        linear_layer.bias.requires_grad = True

    new_class_start = old_num_classes

    def weight_gradient_mask(gradient):
        masked_gradient = gradient.clone()
        masked_gradient[:new_class_start] = 0
        return masked_gradient

    linear_layer.weight.register_hook(weight_gradient_mask)

    if linear_layer.bias is not None:

        def bias_gradient_mask(gradient):
            masked_gradient = gradient.clone()
            masked_gradient[:new_class_start] = 0
            return masked_gradient

        linear_layer.bias.register_hook(bias_gradient_mask)


# =============================================================
# Parameter statistics & Integrity Verification
# =============================================================


def print_trainable_parameters(model) -> None:
    """Print trainable and total parameter counts."""

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)

    total_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Total parameters:     {total_count:,}")


def verify_original_classifier(
    model,
    original_weight: torch.Tensor,
    original_bias: torch.Tensor,
    old_num_classes: int,
) -> None:
    """Verify that original ResNet-18 classifier rows have not changed."""
    _, linear_layer = get_classifier_layer(model)

    with torch.no_grad():
        current_weight = linear_layer.weight[:old_num_classes].detach().cpu()
        original_weight = original_weight[:old_num_classes].detach().cpu()

        weights_unchanged = torch.equal(current_weight, original_weight)

        if linear_layer.bias is not None:
            current_bias = linear_layer.bias[:old_num_classes].detach().cpu()
            original_bias = original_bias[:old_num_classes].detach().cpu()
            bias_unchanged = torch.equal(current_bias, original_bias)
        else:
            bias_unchanged = True

    print("\nOriginal classifier verification:")
    print(f"  Original weights unchanged: {weights_unchanged}")
    print(f"  Original bias unchanged:    {bias_unchanged}")

    if not weights_unchanged or not bias_unchanged:
        raise RuntimeError("Original ResNet-18 classifier weights were modified.")


def collect_local_validation_probabilities(
    model,
    loader,
    device,
    old_num_classes: int,
    local_class_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect full-model probabilities for local plant classes."""

    model.eval()

    true_labels = []
    probability_rows = []

    with torch.no_grad():
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device)

            outputs = model(pixel_values)

            probabilities = torch.softmax(outputs, dim=1)

            local_probabilities = (
                probabilities[
                    :,
                    old_num_classes : old_num_classes + local_class_count,
                ]
                .cpu()
                .numpy()
            )

            local_labels = labels.cpu().numpy() - old_num_classes

            true_labels.extend(local_labels.tolist())
            probability_rows.extend(local_probabilities.tolist())

    return (
        np.asarray(true_labels, dtype=int),
        np.asarray(probability_rows, dtype=float),
    )


def build_validation_results(
    validation_dataset: LocalPlantDataset,
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    local_classes: List[str],
) -> List[Dict[str, object]]:
    """Create per-image validation output rows for CSV export."""

    predicted_labels = probabilities.argmax(axis=1)

    rows = []

    for index, ((image_path, _), true_label, predicted_label) in enumerate(
        zip(
            validation_dataset.samples,
            true_labels,
            predicted_labels,
        )
    ):
        confidence = probabilities[index, predicted_label] * 100

        rows.append(
            {
                "ImageName": image_path.name,
                "ImagePath": str(image_path),
                "TrueLabel": local_classes[int(true_label)],
                "PredictedLabel": local_classes[int(predicted_label)],
                "ConfidencePercent": round(float(confidence), 4),
                "Correct": bool(true_label == predicted_label),
            }
        )

    return rows


def create_training_history(history_data: Dict[str, List[float]]) -> TrainingHistory:
    """Convert FineTuner history dictionaries to TrainingHistory."""

    history = TrainingHistory()

    for index, (
        train_loss,
        train_accuracy,
        val_loss,
        val_accuracy,
    ) in enumerate(
        zip(
            history_data["train_loss"],
            history_data["train_accuracy"],
            history_data["val_loss"],
            history_data["val_accuracy"],
        ),
        start=1,
    ):
        history.add_epoch(
            epoch=index,
            train_loss=train_loss,
            val_loss=val_loss,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
        )

    return history


def save_extended_model_artifacts(
    model,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    local_classes: List[str],
) -> Path:
    """Save the extended torchvision model and class metadata."""

    EXTENDED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = EXTENDED_MODEL_DIR / "plantnet_resnet18_extended.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": len(id2label),
            "id2label": id2label,
            "label2id": label2id,
            "local_classes": local_classes,
        },
        model_path,
    )

    metadata_files = [
        "class_idx_to_species_id.json",
        "plantnet300K_species_id_2_name.json",
        "plantnet300K_metadata.json",
        "plantnet300k_class_mapping.json",
    ]

    for filename in metadata_files:
        source_path = BASE_MODEL_DIR / filename
        target_path = EXTENDED_MODEL_DIR / filename

        if source_path.exists():
            shutil.copy2(source_path, target_path)

    with open(
        EXTENDED_MODEL_DIR / "extended_id2label.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {str(index): label for index, label in id2label.items()},
            file,
            indent=2,
        )

    with open(
        EXTENDED_MODEL_DIR / "extended_label2id.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(label2id, file, indent=2)

    with open(
        EXTENDED_MODEL_DIR / "local_classes.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(local_classes, file, indent=2)

    return model_path


# =============================================================
# Main fine-tuning procedure
# =============================================================


def main():
    """Run ResNet-18 local-plant classifier extension."""

    set_seed(RANDOM_SEED)

    device = get_device()

    print("=" * 70)
    print("ResNet-18 Local Plant Extension")
    print("=" * 70)
    print(f"Device: {device}")

    if not BASE_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Original ResNet-18 model was not found at: {BASE_MODEL_DIR}"
        )

    print("\nCreating PlantNet-300K preprocessing transform...")
    transform = get_plantnet_transform()

    print("Loading original ResNet-18 model...")
    model = load_plantnet_model()

    local_classes = discover_local_classes()
    validate_dataset_classes(local_classes)

    print("\nLocal plant classes:")
    for class_name in local_classes:
        print(f"  - {class_name}")

    _, linear_layer = get_classifier_layer(model)

    original_classifier_weight = linear_layer.weight.detach().clone()

    original_classifier_bias = (
        linear_layer.bias.detach().clone()
        if linear_layer.bias is not None
        else torch.empty(0)
    )

    print("\nExpanding ResNet-18 classifier...")

    (
        id2label,
        label2id,
        old_num_classes,
    ) = expand_classifier(
        model=model,
        local_classes=local_classes,
    )

    print("\nFreezing pretrained ResNet-18 backbone " "and original classifier rows...")

    freeze_model_except_new_classes(
        model=model,
        old_num_classes=old_num_classes,
    )

    print_trainable_parameters(model)

    _, current_linear = get_classifier_layer(model)

    expected_new_parameters = len(local_classes) * current_linear.in_features

    if current_linear.bias is not None:
        expected_new_parameters += len(local_classes)

    print(f"Expected new-class parameters: " f"{expected_new_parameters:,}")

    print("\nLoading datasets...")

    train_dataset = LocalPlantDataset(
        root_dir=TRAIN_DIR,
        transform=transform,
        label2id=label2id,
    )

    validation_dataset = LocalPlantDataset(
        root_dir=VALIDATION_DIR,
        transform=transform,
        label2id=label2id,
    )

    print(f"Training images:   {len(train_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fine_tune_config = FineTuneConfig(
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        freeze_backbone=True,
        experiment_dir=RESULTS_DIR / "checkpoints",
        output_model_dir=EXTENDED_MODEL_DIR,
        save_best_only=True,
    )

    adapter = TorchVisionModelAdapter(
        model=model,
        keep_batchnorm_eval=True,
    )

    adapter.to(device)

    fine_tuner = FineTuner(
        adapter=adapter,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=fine_tune_config,
        device=device,
    )

    print("\nStarting training...")
    print("-" * 70)

    history_data = fine_tuner.train()

    # -------------------------------------------------------------
    # Load the best checkpoint produced by FineTuner.
    # -------------------------------------------------------------

    best_checkpoint = fine_tuner.best_checkpoint_path

    if best_checkpoint is None:
        raise RuntimeError(
            "Fine-tuning completed without producing " "a best-model checkpoint."
        )

    print(f"\nLoading best checkpoint:\n" f"  {best_checkpoint}")

    checkpoint = torch.load(
        best_checkpoint,
        map_location=device,
    )

    adapter.get_model().load_state_dict(checkpoint["model_state_dict"])

    best_checkpoint_epoch = checkpoint.get("epoch")

    best_checkpoint_accuracy = checkpoint.get("validation_accuracy")

    print(f"Best checkpoint epoch: " f"{best_checkpoint_epoch}")

    if best_checkpoint_accuracy is not None:
        print(
            f"Best checkpoint validation accuracy: " f"{best_checkpoint_accuracy:.4f}"
        )

    # -------------------------------------------------------------
    # Convert training history for analytics.
    # -------------------------------------------------------------

    history = create_training_history(history_data)

    best_validation_accuracy = max(history.val_accuracy)

    best_epoch = history.val_accuracy.index(best_validation_accuracy) + 1

    # -------------------------------------------------------------
    # Generate fine-tuning analytics.
    # -------------------------------------------------------------

    print("\nGenerating fine-tuning analytics...")

    export_training_history(
        history=history,
        output_dir=str(RESULTS_DIR),
        output_filename="resnet18_training_history.csv",
    )

    plot_training_loss(
        history=history,
        output_path=str(RESULTS_DIR / "resnet18_training_validation_loss.png"),
    )

    plot_training_accuracy(
        history=history,
        output_path=str(RESULTS_DIR / "resnet18_training_validation_accuracy.png"),
    )

    # -------------------------------------------------------------
    # Calculate local-class ROC-AUC.
    # -------------------------------------------------------------

    (
        validation_true_labels,
        validation_probabilities,
    ) = collect_local_validation_probabilities(
        model=model,
        loader=validation_loader,
        device=device,
        old_num_classes=old_num_classes,
        local_class_count=len(local_classes),
    )

    roc_auc = print_roc_auc(
        true_labels=validation_true_labels,
        probabilities=validation_probabilities,
    )

    plotted_roc_auc = plot_roc_auc_curve(
        true_labels=validation_true_labels,
        probabilities=validation_probabilities,
        class_names=local_classes,
        output_path=str(RESULTS_DIR / "resnet18_validation_roc_auc.png"),
    )

    # Use the plotted/calculated value for the summary
    # when it is available.
    if plotted_roc_auc is not None:
        roc_auc = plotted_roc_auc

    # -------------------------------------------------------------
    # Build validation prediction results.
    # -------------------------------------------------------------

    validation_results = build_validation_results(
        validation_dataset=validation_dataset,
        true_labels=validation_true_labels,
        probabilities=validation_probabilities,
        local_classes=local_classes,
    )

    export_evaluation_results(
        evaluation_results=validation_results,
        output_dir=str(RESULTS_DIR),
        output_filename="resnet18_validation_predictions.csv",
    )

    # -------------------------------------------------------------
    # Export experiment summary.
    # -------------------------------------------------------------

    export_evaluation_summary(
        evaluation_summary=[
            {
                "ModelName": "ResNet18 Extended",
                "Epochs": EPOCHS,
                "BestEpoch": best_epoch,
                "BestValidationAccuracy": round(
                    best_validation_accuracy,
                    6,
                ),
                "ValidationRocAuc": (
                    round(roc_auc, 6) if roc_auc is not None else None
                ),
                "TrainingImages": len(train_dataset),
                "ValidationImages": len(validation_dataset),
                "LocalClasses": len(local_classes),
            }
        ],
        output_dir=str(RESULTS_DIR),
        output_filename="resnet18_training_summary.csv",
    )

    # -------------------------------------------------------------
    # Verify that the original PlantNet classifier rows
    # were not modified.
    # -------------------------------------------------------------

    verify_original_classifier(
        model=model,
        original_weight=original_classifier_weight,
        original_bias=original_classifier_bias,
        old_num_classes=old_num_classes,
    )

    # -------------------------------------------------------------
    # Save the final extended model.
    # -------------------------------------------------------------

    print("\nSaving extended ResNet-18 model...")

    save_extended_model_artifacts(
        model=model,
        id2label=id2label,
        label2id=label2id,
        local_classes=local_classes,
    )

    print("-" * 70)
    print("Fine-tuning completed.")
    print(f"Best validation accuracy: " f"{best_validation_accuracy:.4f}")
    print(f"Saved model: " f"{EXTENDED_MODEL_DIR}")
    print(f"Saved analytics: " f"{RESULTS_DIR}")
    print("\nThe original ResNet-18 model was not modified.")


if __name__ == "__main__":
    main()
