# core/analytics.py
"""
General analytics and metrics utilities.

This module provides reusable analytics for model comparison and training,
including:

- Prediction confidence logging
- Confidence comparison charts
- Training/validation loss tracking
- Training/validation accuracy tracking
- Training history CSV export
- Loss and accuracy plots
- Binary and multiclass ROC-AUC calculation
"""

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


# ---------------------------------------------------------------------------
# Prediction Analytics
# ---------------------------------------------------------------------------


def truncate_filename(filename: str, max_len: int = 15) -> str:
    """
    Truncates long filenames while preserving extension.
    Example: '00_Lansium domesticum.jpg' -> '00_Lansi...jpg'
    """
    if not isinstance(filename, str) or len(filename) <= max_len:
        return filename

    name, ext = os.path.splitext(filename)
    # Reserve space for "..." and extension
    available_len = max_len - len(ext) - 3
    if available_len < 1:
        return filename[: max_len - 3] + "..."

    return f"{name[:available_len]}...{ext}"


def log_prediction_result(
    image_name,
    model_name,
    species,
    confidence,
    log_dir="./results",
):
    """Append prediction details to the historical evaluation log."""

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir,
        "model_confidence_history.csv",
    )

    file_exists = os.path.exists(log_file)

    with open(
        log_file,
        mode="a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(
                [
                    "Timestamp",
                    "ImageName",
                    "ModelName",
                    "PredictedSpecies",
                    "ConfidencePercent",
                ]
            )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        writer.writerow(
            [
                timestamp,
                image_name,
                model_name,
                species,
                f"{confidence:.2f}",
            ]
        )


def plot_confidence_trends(
    log_path="./results/model_confidence_history.csv",
):
    """
    Generate a grouped bar chart comparing model confidence
    for each evaluated image.
    """

    if not os.path.exists(log_path):
        print("No historical logs found yet. " "Run some predictions first!")
        return

    df = pd.read_csv(log_path)

    if df.empty:
        print("Historical log is empty.")
        return

    # Clean numeric data strictly
    df["ConfidencePercent"] = pd.to_numeric(
        df["ConfidencePercent"],
        errors="coerce",
    )

    print("\n--- Model Confidence Statistics ---")

    # Groupby stats on filtered numeric series
    stats = df.groupby("ModelName")["ConfidencePercent"].agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
            "std",
        ]
    )

    print(stats.to_string())
    print("-" * 40)

    # -------------------------------
    # Truncate Long Image Names
    # -------------------------------
    df["ShortImageName"] = df["ImageName"].apply(
        lambda x: truncate_filename(str(x), max_len=16)
    )

    # Group by short image name and model.
    plot_df = df.pivot_table(
        index="ShortImageName",
        columns="ModelName",
        values="ConfidencePercent",
        aggfunc="mean",
    )

    # Rename index name to prevent string metric collisions
    plot_df.index.name = "Image"

    # -------------------------------
    # Grouped bar chart
    # -------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 6),
        constrained_layout=True,
    )

    plot_df.plot(
        kind="bar",
        ax=ax,
        width=0.8,
    )

    ax.set_title(
        "Model Confidence Comparison by Image",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Image", fontsize=11, fontweight="bold")
    ax.set_ylabel("Confidence (%)", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 100)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.6,
    )

    # Place legend outside plot box.
    ax.legend(
        title="Models",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.xticks(rotation=30, ha="right", fontsize=9)

    chart_path = "./results/confidence_comparison_chart.png"

    fig.savefig(
        chart_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Confidence comparison chart " f"successfully saved to {chart_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Training History
# ---------------------------------------------------------------------------


class TrainingHistory:
    """
    Store training and validation metrics for each epoch.

    This class is independent of the model architecture and can therefore
    be reused by Juppy44, PlantNet, CNN, ResNet, ViT, or other models.
    """

    def __init__(self):
        self.epochs: List[int] = []
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.train_accuracy: List[float] = []
        self.val_accuracy: List[float] = []

    def add_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_accuracy: float,
        val_accuracy: float,
    ):
        """Record metrics for one training epoch."""

        self.epochs.append(epoch)

        self.train_loss.append(float(train_loss))

        self.val_loss.append(float(val_loss))

        self.train_accuracy.append(float(train_accuracy))

        self.val_accuracy.append(float(val_accuracy))

    def to_dict(self) -> Dict[str, List[float]]:
        """Return the recorded metrics as a dictionary."""

        return {
            "epoch": self.epochs,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
        }


"""
    def save_csv(self, output_path: str):
        
        # Save the complete training history as a CSV file.
        

        directory = os.path.dirname(output_path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        with open(
            output_path,
            mode="w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "train_accuracy",
                    "val_accuracy",
                ]
            )

            for row in zip(
                self.epochs,
                self.train_loss,
                self.val_loss,
                self.train_accuracy,
                self.val_accuracy,
            ):
                writer.writerow(row)

        print(
            f"Training history saved to "
            f"{output_path}"
        )
"""

# ---------------------------------------------------------------------------
# Training Visualization
# ---------------------------------------------------------------------------


def plot_training_loss(
    history: TrainingHistory,
    output_path: Optional[str] = None,
):
    """
    Plot training and validation loss.

    A common indication of overfitting is training loss continuing
    to decrease while validation loss begins to increase.
    """

    if not history.epochs:
        print("No training history available.")
        return

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.epochs,
        history.train_loss,
        label="Training Loss",
    )

    plt.plot(
        history.epochs,
        history.val_loss,
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if output_path:
        directory = os.path.dirname(output_path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Training loss graph saved to " f"{output_path}")

    plt.close()


def plot_training_accuracy(
    history: TrainingHistory,
    output_path: Optional[str] = None,
):
    """
    Plot training and validation accuracy.

    A growing difference between training and validation accuracy
    can indicate overfitting.
    """

    if not history.epochs:
        print("No training history available.")
        return

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.epochs,
        history.train_accuracy,
        label="Training Accuracy",
    )

    plt.plot(
        history.epochs,
        history.val_accuracy,
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if output_path:
        directory = os.path.dirname(output_path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Training accuracy graph saved to " f"{output_path}")

    plt.close()


# ---------------------------------------------------------------------------
# ROC-AUC
# ---------------------------------------------------------------------------


def calculate_roc_auc(
    true_labels: Sequence[int],
    probabilities: np.ndarray,
    multi_class: str = "ovr",
) -> Optional[float]:
    """
    Calculate ROC-AUC for classification predictions.

    Parameters
    ----------
    true_labels:
        Ground-truth class indices.

    probabilities:
        Model probabilities with shape:

            (number_of_samples, number_of_classes)

    multi_class:
        Multiclass ROC-AUC strategy.

        "ovr" = one-vs-rest
        "ovo" = one-vs-one

    Returns
    -------
    float or None
        ROC-AUC score.

        Returns None when ROC-AUC cannot be calculated, such as
        when the validation data contains only one class.
    """

    true_labels = np.asarray(true_labels)

    probabilities = np.asarray(probabilities)

    if len(true_labels) == 0:
        return None

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape " "(samples, classes).")

    if probabilities.shape[0] != len(true_labels):
        raise ValueError(
            "Number of probability rows must match " "the number of true labels."
        )

    unique_classes = np.unique(true_labels)

    # ROC-AUC cannot be calculated from a
    # validation set containing only one class.
    if len(unique_classes) < 2:
        return None

    try:
        # Binary classification.
        if probabilities.shape[1] == 2:
            return float(
                roc_auc_score(
                    true_labels,
                    probabilities[:, 1],
                )
            )

        # Multiclass classification.
        return float(
            roc_auc_score(
                true_labels,
                probabilities,
                multi_class=multi_class,
                average="macro",
            )
        )

    except ValueError:
        return None


def print_roc_auc(
    true_labels: Sequence[int],
    probabilities: np.ndarray,
    multi_class: str = "ovr",
) -> Optional[float]:
    """
    Calculate and print ROC-AUC.

    Returns the calculated value, or None when ROC-AUC
    cannot be calculated.
    """

    roc_auc = calculate_roc_auc(
        true_labels=true_labels,
        probabilities=probabilities,
        multi_class=multi_class,
    )

    if roc_auc is None:
        print("ROC-AUC: not available " "(insufficient class information).")
    else:
        print(f"ROC-AUC: {roc_auc:.4f}")

    return roc_auc


def plot_roc_auc_curve(
    true_labels: Sequence[int],
    probabilities: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    output_path: Optional[str] = None,
    multi_class: str = "ovr",
) -> Optional[float]:
    """
    Plot ROC curve information and return macro ROC-AUC.

    For binary classification, the positive class curve is plotted.
    For multiclass classification, each class curve plus a macro-average
    curve are plotted.
    """

    true_labels = np.asarray(true_labels)
    probabilities = np.asarray(probabilities)

    roc_auc = calculate_roc_auc(
        true_labels=true_labels,
        probabilities=probabilities,
        multi_class=multi_class,
    )

    if roc_auc is None:
        print("ROC-AUC graph skipped " "(insufficient class information).")
        return None

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape " "(samples, classes).")

    class_count = probabilities.shape[1]

    if class_names is None:
        class_names = [f"Class {index}" for index in range(class_count)]

    plt.figure(figsize=(8, 6))

    if class_count == 2:
        fpr, tpr, _ = roc_curve(
            true_labels,
            probabilities[:, 1],
        )

        plt.plot(
            fpr,
            tpr,
            label=f"ROC-AUC = {roc_auc:.4f}",
        )
    else:
        binarized_labels = label_binarize(
            true_labels,
            classes=list(range(class_count)),
        )

        all_fpr = np.unique(
            np.concatenate(
                [
                    roc_curve(
                        binarized_labels[:, class_index],
                        probabilities[:, class_index],
                    )[0]
                    for class_index in range(class_count)
                ]
            )
        )

        mean_tpr = np.zeros_like(all_fpr)

        for class_index in range(class_count):
            fpr, tpr, _ = roc_curve(
                binarized_labels[:, class_index],
                probabilities[:, class_index],
            )

            mean_tpr += np.interp(
                all_fpr,
                fpr,
                tpr,
            )

            class_auc = auc(
                fpr,
                tpr,
            )

            plt.plot(
                fpr,
                tpr,
                linewidth=1,
                alpha=0.5,
                label=f"{class_names[class_index]} ({class_auc:.3f})",
            )

        mean_tpr /= class_count

        plt.plot(
            all_fpr,
            mean_tpr,
            color="black",
            linewidth=2,
            label=f"Macro-average ({roc_auc:.4f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1,
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Validation ROC-AUC")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(True)
    plt.tight_layout()

    if output_path:
        directory = os.path.dirname(output_path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"ROC-AUC graph saved to " f"{output_path}")

    plt.close()

    return roc_auc
