"""
Fine-tune Juppy44 (Vit based) with additional local plant classes.

This script extends the original Juppy44 classification head while
preserving the pretrained classifier weights for the original classes.

Only the newly added local-plant classifier rows are trained.
The original Juppy44 backbone and original classifier rows remain frozen.

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
    weights/juppy44_extended/
        config.json
        model.safetensors
        preprocessor_config.json
"""

# fine_tune/juppy44_fineTuner.py

import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)
from core.analytics import (
    TrainingHistory,
    plot_roc_auc_curve,
    plot_training_accuracy,
    plot_training_loss,
    print_roc_auc,
)
from core.formatters import (
    export_evaluation_results,
    export_evaluation_summary,
    export_training_history,
)

# =============================================================
# Configuration
# =============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

BASE_MODEL_DIR = ROOT_DIR / "weights" / "juppy44"

EXTENDED_MODEL_DIR = ROOT_DIR / "weights" / "juppy44_extended"

RESULTS_DIR = ROOT_DIR / "results" / "fine_tune" / "juppy44_extended"

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

    Directory structure:

        split/
            plant_a/
                image1.jpg
                image2.jpg
            plant_b/
                image1.jpg
                image2.jpg
    """

    def __init__(
        self,
        root_dir: Path,
        processor,
        label2id: Dict[str, int],
    ):
        self.root_dir = Path(root_dir)

        self.processor = processor

        self.label2id = label2id

        self.samples: List[Tuple[Path, int]] = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: " f"{self.root_dir}")

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
                    f"Unknown class '{class_name}' " f"found in {self.root_dir}"
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

        encoded = self.processor(
            images=image,
            return_tensors="pt",
        )

        pixel_values = encoded["pixel_values"].squeeze(0)

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


def get_existing_labels(
    model,
) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Read the original Juppy44 class mappings.
    """

    id2label = {int(index): label for index, label in model.config.id2label.items()}

    label2id = {label: int(index) for label, index in model.config.label2id.items()}

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


def validate_dataset_classes(
    local_classes: List[str],
) -> None:
    """
    Make sure validation contains the same local classes
    as the training dataset.
    """

    if not VALIDATION_DIR.exists():
        raise FileNotFoundError(f"Validation directory not found: " f"{VALIDATION_DIR}")

    validation_classes = sorted(
        directory.name for directory in VALIDATION_DIR.iterdir() if directory.is_dir()
    )

    missing_classes = sorted(set(local_classes) - set(validation_classes))

    if missing_classes:
        raise ValueError(
            "The following training classes are missing "
            "from validation:\n"
            f"{missing_classes}"
        )


# =============================================================
# Classifier expansion
# =============================================================


def expand_classifier(
    model,
    local_classes: List[str],
) -> Tuple[
    Dict[int, str],
    Dict[str, int],
    int,
]:
    """
    Expand the Juppy44 classifier with local plant classes.

    Existing classifier weights are copied unchanged.

    New classifier rows are initialized separately.

    Returns:
        id2label:
            Updated class index -> label mapping.

        label2id:
            Updated label -> class index mapping.

        old_num_classes:
            Number of original Juppy44 classes.
    """

    old_id2label, old_label2id = get_existing_labels(model)

    old_num_classes = len(old_id2label)

    # ---------------------------------------------------------
    # Check duplicate classes.
    # ---------------------------------------------------------

    duplicate_classes = [
        class_name for class_name in local_classes if class_name in old_label2id
    ]

    if duplicate_classes:
        raise ValueError(
            "The following local classes already exist "
            "in Juppy44: "
            f"{duplicate_classes}"
        )

    # ---------------------------------------------------------
    # Check classifier type.
    # ---------------------------------------------------------

    old_classifier = model.classifier

    if not isinstance(
        old_classifier,
        nn.Linear,
    ):
        raise TypeError(
            "Expected Juppy44 classifier to be "
            "torch.nn.Linear, but found: "
            f"{type(old_classifier)}"
        )

    old_num_classes_from_layer = old_classifier.out_features

    if old_num_classes_from_layer != old_num_classes:
        raise ValueError(
            "Model configuration and classifier "
            "disagree about the number of classes: "
            f"config={old_num_classes}, "
            f"classifier={old_num_classes_from_layer}"
        )

    # ---------------------------------------------------------
    # Create new label mappings.
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Create expanded classifier.
    # ---------------------------------------------------------

    new_classifier = nn.Linear(
        old_classifier.in_features,
        new_num_classes,
        bias=old_classifier.bias is not None,
    )

    # ---------------------------------------------------------
    # Copy original classifier weights.
    # ---------------------------------------------------------

    with torch.no_grad():

        new_classifier.weight[:old_num_classes].copy_(old_classifier.weight)

        if old_classifier.bias is not None:

            new_classifier.bias[:old_num_classes].copy_(old_classifier.bias)

    # ---------------------------------------------------------
    # Replace classifier.
    # ---------------------------------------------------------

    model.classifier = new_classifier

    # ---------------------------------------------------------
    # Update Hugging Face configuration.
    # ---------------------------------------------------------

    model.config.num_labels = new_num_classes

    model.config.id2label = {int(index): label for index, label in new_id2label.items()}

    model.config.label2id = {label: int(index) for label, index in new_label2id.items()}

    return (
        new_id2label,
        new_label2id,
        old_num_classes,
    )


# =============================================================
# Freeze model
# =============================================================


def freeze_model_except_new_classes(
    model,
    old_num_classes: int,
) -> None:
    """
    Freeze the complete pretrained Juppy44 model.

    Only the newly added classifier rows are allowed
    to receive gradient updates.

    The original classifier rows remain unchanged.
    """

    # ---------------------------------------------------------
    # Freeze everything first.
    # ---------------------------------------------------------

    for parameter in model.parameters():
        parameter.requires_grad = False

    # ---------------------------------------------------------
    # The classifier itself must be trainable because
    # PyTorch parameters cannot independently freeze individual
    # rows of a tensor.
    # ---------------------------------------------------------

    model.classifier.weight.requires_grad = True

    if model.classifier.bias is not None:
        model.classifier.bias.requires_grad = True

    # ---------------------------------------------------------
    # Save the original classifier boundary.
    # ---------------------------------------------------------

    new_class_start = old_num_classes

    # ---------------------------------------------------------
    # Mask gradients for the original classifier rows.
    #
    # The new rows receive gradients.
    # The original rows receive zero gradients.
    # ---------------------------------------------------------

    def weight_gradient_mask(gradient):
        masked_gradient = gradient.clone()

        masked_gradient[:new_class_start] = 0

        return masked_gradient

    model.classifier.weight.register_hook(weight_gradient_mask)

    if model.classifier.bias is not None:

        def bias_gradient_mask(gradient):
            masked_gradient = gradient.clone()

            masked_gradient[:new_class_start] = 0

            return masked_gradient

        model.classifier.bias.register_hook(bias_gradient_mask)


# =============================================================
# Parameter statistics
# =============================================================


def print_trainable_parameters(
    model,
) -> None:
    """Print trainable and total parameter counts."""

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)

    total_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Trainable parameters: " f"{trainable_count:,}")

    print(f"Total parameters:     " f"{total_count:,}")


# =============================================================
# Classifier integrity
# =============================================================


def save_original_classifier(
    model,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Save a copy of the original classifier rows.

    These are later compared with the trained model to verify
    that the original Juppy44 classifier was preserved.
    """

    with torch.no_grad():

        original_weight = model.classifier.weight.detach().clone()

        if model.classifier.bias is not None:

            original_bias = model.classifier.bias.detach().clone()

        else:

            original_bias = torch.empty(0)

    return (
        original_weight,
        original_bias,
    )


def verify_original_classifier(
    model,
    original_weight: torch.Tensor,
    original_bias: torch.Tensor,
    old_num_classes: int,
) -> None:
    """
    Verify that original Juppy44 classifier rows
    have not changed.
    """

    with torch.no_grad():

        current_weight = model.classifier.weight[:old_num_classes].detach().cpu()

        original_weight = original_weight[:old_num_classes].detach().cpu()

        weights_unchanged = torch.equal(
            current_weight,
            original_weight,
        )

        if model.classifier.bias is not None:

            current_bias = model.classifier.bias[:old_num_classes].detach().cpu()

            original_bias = original_bias[:old_num_classes].detach().cpu()

            bias_unchanged = torch.equal(
                current_bias,
                original_bias,
            )

        else:

            bias_unchanged = True

    print("\nOriginal classifier verification:")

    print(f"  Original weights unchanged: " f"{weights_unchanged}")

    print(f"  Original bias unchanged:    " f"{bias_unchanged}")

    if not weights_unchanged or not bias_unchanged:

        raise RuntimeError("Original Juppy44 classifier weights " "were modified.")


# =============================================================
# Training
# =============================================================


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
) -> Tuple[float, float]:

    model.train()

    total_loss = 0.0

    correct = 0

    total = 0

    for pixel_values, labels in loader:

        pixel_values = pixel_values.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values=pixel_values)

        loss = criterion(
            outputs.logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += batch_size

    average_loss = total_loss / total

    accuracy = correct / total

    return (
        average_loss,
        accuracy,
    )


# =============================================================
# Validation
# =============================================================


def validate(
    model,
    loader,
    criterion,
    device,
) -> Tuple[float, float]:

    model.eval()

    total_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for pixel_values, labels in loader:

            pixel_values = pixel_values.to(device)

            labels = labels.to(device)

            outputs = model(pixel_values=pixel_values)

            loss = criterion(
                outputs.logits,
                labels,
            )

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += batch_size

    average_loss = total_loss / total

    accuracy = correct / total

    return (
        average_loss,
        accuracy,
    )


def collect_local_validation_probabilities(
    model,
    loader,
    device,
    old_num_classes: int,
    local_class_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect validation labels and probabilities for the local classes only.

    The extended classifier still contains the original Juppy44 labels.
    ROC-AUC for fine-tuning should compare only the added local classes.
    """

    model.eval()

    true_labels = []

    probability_rows = []

    with torch.no_grad():

        for pixel_values, labels in loader:

            pixel_values = pixel_values.to(device)

            outputs = model(pixel_values=pixel_values)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1,
            )

            local_probabilities = probabilities[
                :,
                old_num_classes : old_num_classes + local_class_count,
            ]

            local_probabilities = local_probabilities.cpu().numpy()

            row_sums = local_probabilities.sum(
                axis=1,
                keepdims=True,
            )

            local_probabilities = np.divide(
                local_probabilities,
                row_sums,
                out=np.zeros_like(local_probabilities),
                where=row_sums > 0,
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


# =============================================================
# Main fine-tuning procedure
# =============================================================


def main():
    """Run Juppy44 local-plant classifier extension."""

    set_seed(RANDOM_SEED)

    device = get_device()

    print("=" * 70)

    print("Juppy44 Local Plant Extension")

    print("=" * 70)

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # 1. Check original model
    # ---------------------------------------------------------

    if not BASE_MODEL_DIR.exists():

        raise FileNotFoundError(
            "Original Juppy44 model was not found at: " f"{BASE_MODEL_DIR}"
        )

    # ---------------------------------------------------------
    # 2. Load original processor
    # ---------------------------------------------------------

    print("\nLoading original Juppy44 processor...")

    processor = AutoImageProcessor.from_pretrained(
        BASE_MODEL_DIR,
        local_files_only=True,
    )

    # ---------------------------------------------------------
    # 3. Load original Juppy44 model
    # ---------------------------------------------------------

    print("Loading original Juppy44 model...")

    model = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL_DIR,
        local_files_only=True,
    )

    # ---------------------------------------------------------
    # 4. Discover local classes
    # ---------------------------------------------------------

    local_classes = discover_local_classes()

    validate_dataset_classes(local_classes)

    print("\nLocal plant classes:")

    for class_name in local_classes:

        print(f"  - {class_name}")

    # ---------------------------------------------------------
    # 5. Save original classifier
    # ---------------------------------------------------------

    original_classifier_weight = model.classifier.weight.detach().clone()

    if model.classifier.bias is not None:

        original_classifier_bias = model.classifier.bias.detach().clone()

    else:

        original_classifier_bias = torch.empty(0)

    # ---------------------------------------------------------
    # 6. Expand classifier
    # ---------------------------------------------------------

    print("\nExpanding Juppy44 classifier...")

    (
        id2label,
        label2id,
        old_num_classes,
    ) = expand_classifier(
        model=model,
        local_classes=local_classes,
    )

    # ---------------------------------------------------------
    # 7. Freeze original model and classifier rows
    # ---------------------------------------------------------

    print("\nFreezing pretrained Juppy44 " "backbone and original classifier rows...")

    freeze_model_except_new_classes(
        model=model,
        old_num_classes=old_num_classes,
    )

    print_trainable_parameters(model)

    expected_new_parameters = len(local_classes) * model.classifier.in_features

    if model.classifier.bias is not None:

        expected_new_parameters += len(local_classes)

    print(f"Expected new-class parameters: " f"{expected_new_parameters:,}")

    # ---------------------------------------------------------
    # 8. Create datasets
    # ---------------------------------------------------------

    print("\nLoading datasets...")

    train_dataset = LocalPlantDataset(
        root_dir=TRAIN_DIR,
        processor=processor,
        label2id=label2id,
    )

    validation_dataset = LocalPlantDataset(
        root_dir=VALIDATION_DIR,
        processor=processor,
        label2id=label2id,
    )

    print(f"Training images:   " f"{len(train_dataset)}")

    print(f"Validation images: " f"{len(validation_dataset)}")

    # ---------------------------------------------------------
    # 9. Create data loaders
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 10. Move model to device
    # ---------------------------------------------------------

    model.to(device)

    # ---------------------------------------------------------
    # 11. Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 12. Training loop
    # ---------------------------------------------------------

    history = TrainingHistory()

    best_validation_accuracy = -1.0

    best_epoch = 0

    best_state_dict = None

    print("\nStarting training...")

    print("-" * 70)

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        (
            train_loss,
            train_accuracy,
        ) = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        (
            validation_loss,
            validation_accuracy,
        ) = validate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_accuracy:.4f} | "
            f"val loss: {validation_loss:.4f} | "
            f"val acc: {validation_accuracy:.4f}"
        )

        history.add_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=validation_loss,
            train_accuracy=train_accuracy,
            val_accuracy=validation_accuracy,
        )

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy

            best_epoch = epoch

            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    # ---------------------------------------------------------
    # 13. Restore best model
    # ---------------------------------------------------------

    if best_state_dict is not None:

        model.load_state_dict(best_state_dict)

    # ---------------------------------------------------------
    # 14. Export analytics and validation outputs
    # ---------------------------------------------------------

    print("\nGenerating fine-tuning analytics...")

    export_training_history(
        history=history,
        output_dir=str(RESULTS_DIR),
        output_filename="juppy44_training_history.csv",
    )

    plot_training_loss(
        history=history,
        output_path=str(RESULTS_DIR / "juppy44_training_validation_loss.png"),
    )

    plot_training_accuracy(
        history=history,
        output_path=str(RESULTS_DIR / "juppy44_training_validation_accuracy.png"),
    )

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

    roc_auc = plot_roc_auc_curve(
        true_labels=validation_true_labels,
        probabilities=validation_probabilities,
        class_names=local_classes,
        output_path=str(RESULTS_DIR / "juppy44_validation_roc_auc.png"),
    )

    validation_results = build_validation_results(
        validation_dataset=validation_dataset,
        true_labels=validation_true_labels,
        probabilities=validation_probabilities,
        local_classes=local_classes,
    )

    export_evaluation_results(
        evaluation_results=validation_results,
        output_dir=str(RESULTS_DIR),
        output_filename="juppy44_validation_predictions.csv",
    )

    export_evaluation_summary(
        evaluation_summary=[
            {
                "ModelName": "Juppy44 Extended",
                "Epochs": EPOCHS,
                "BestEpoch": best_epoch,
                "BestValidationAccuracy": round(
                    best_validation_accuracy,
                    6,
                ),
                "ValidationRocAuc": (
                    round(
                        roc_auc,
                        6,
                    )
                    if roc_auc is not None
                    else None
                ),
                "TrainingImages": len(train_dataset),
                "ValidationImages": len(validation_dataset),
                "LocalClasses": len(local_classes),
            }
        ],
        output_dir=str(RESULTS_DIR),
        output_filename="juppy44_training_summary.csv",
    )

    # ---------------------------------------------------------
    # 15. Verify original classifier
    # ---------------------------------------------------------

    verify_original_classifier(
        model=model,
        original_weight=original_classifier_weight,
        original_bias=original_classifier_bias,
        old_num_classes=old_num_classes,
    )

    # ---------------------------------------------------------
    # 16. Create output directory
    # ---------------------------------------------------------

    EXTENDED_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 17. Save extended model
    # ---------------------------------------------------------

    print("\nSaving extended Juppy44 model...")

    model.save_pretrained(
        EXTENDED_MODEL_DIR,
        safe_serialization=True,
    )

    # ---------------------------------------------------------
    # 18. Save processor
    # ---------------------------------------------------------

    processor.save_pretrained(EXTENDED_MODEL_DIR)

    # ---------------------------------------------------------
    # 19. Final information
    # ---------------------------------------------------------

    print("-" * 70)

    print("Fine-tuning completed.")

    print(f"Best validation accuracy: " f"{best_validation_accuracy:.4f}")

    print(f"Saved model: " f"{EXTENDED_MODEL_DIR}")

    print(f"Saved analytics: " f"{RESULTS_DIR}")

    print("\nThe original Juppy44 model was not modified.")


if __name__ == "__main__":
    main()
