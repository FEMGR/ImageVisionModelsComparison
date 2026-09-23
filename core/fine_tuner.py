"""
Generic PyTorch image-classification fine-tuning engine.
"""

# core/fine_tuner.py

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.fine_tuning import FineTuneConfig
from core.model_adapter import VisionModelAdapter


@dataclass
class EpochResult:
    """Results produced by one training or validation epoch."""

    loss: float
    accuracy: float


class FineTuner:
    """
    Model-agnostic fine-tuning engine.

    Architecture-specific operations are delegated to a
    VisionModelAdapter.
    """

    def __init__(
        self,
        adapter: VisionModelAdapter,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        config: FineTuneConfig,
        device: torch.device,
    ):
        self.adapter = adapter
        self.model = adapter.get_model()

        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.config = config
        self.device = device

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(
            adapter.trainable_parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def train(self) -> Dict[str, list]:
        """Run the complete fine-tuning experiment."""

        self.config.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        best_accuracy = 0.0

        for epoch in range(1, self.config.epochs + 1):

            train_result = self._run_epoch(
                self.train_loader,
                training=True,
            )

            val_result = self._run_epoch(
                self.validation_loader,
                training=False,
            )

            history["train_loss"].append(train_result.loss)
            history["train_accuracy"].append(train_result.accuracy)

            history["val_loss"].append(val_result.loss)
            history["val_accuracy"].append(val_result.accuracy)

            print(
                f"Epoch {epoch}/{self.config.epochs} | "
                f"train loss={train_result.loss:.4f} | "
                f"train acc={train_result.accuracy:.4f} | "
                f"val loss={val_result.loss:.4f} | "
                f"val acc={val_result.accuracy:.4f}"
            )

            if val_result.accuracy > best_accuracy:
                best_accuracy = val_result.accuracy

                self._save_checkpoint(
                    epoch=epoch,
                    validation_accuracy=val_result.accuracy,
                )

        return history

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
    ) -> EpochResult:

        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            if training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    labels,
                )

                if training:
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

        return EpochResult(
            loss=total_loss / total,
            accuracy=correct / total,
        )

    def _save_checkpoint(
        self,
        epoch: int,
        validation_accuracy: float,
    ) -> Path:

        path = self.config.output_dir / "best_model.pt"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "validation_accuracy": validation_accuracy,
            },
            path,
        )

        return path
