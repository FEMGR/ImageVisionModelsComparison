"""
Every plant identification model inherits from this class so the main
evaluation pipeline does not need to know how each specific model works.
"""

# models/base_model.py

from abc import ABC, abstractmethod
from typing import List, Tuple


class BasePlantModel(ABC):
    """Abstract interface for plant identification models."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def load(self, model_path: str = None):
        """
        Load the model into memory.

        Args:
            model_path:
                Optional path to custom model weights.
        """
        pass

    @abstractmethod
    def predict(
        self,
        image_path: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Predict the most likely plant species.

        Predictions must be returned in descending confidence order.

        Args:
            image_path:
                Path to the input image.

            top_k:
                Maximum number of predictions to return.

        Returns:
            List of tuples in the form:

                [
                    ("Species A", 89.64),
                    ("Species B", 7.21),
                    ("Species C", 2.13),
                    ...
                ]

            Confidence values are percentages from 0 to 100.
        """
        pass
