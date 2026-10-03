"""
Configuration for vision-model fine-tuning experiments.
"""

# core/fine_tuning.py

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FineTuneConfig:
    """Configuration for a fine-tuning experiment."""

    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    freeze_backbone: bool = True

    experiment_dir: Path = Path("artifacts/plant_identification/fine_tuning")

    output_model_dir: Path = Path("weights/juppy44_extended")
    save_best_only: bool = True

    @property
    def output_dir(self) -> Path:
        """Directory used by the generic fine-tuner for checkpoints."""

        return self.experiment_dir
