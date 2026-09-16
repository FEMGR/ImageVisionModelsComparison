"""
Every model will inherit from this class so the main script
doesn't need to know how the specific model works under the hood.
"""
#models/base_model.py

from abc import ABC, abstractmethod

class BasePlantModel(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def load(self, model_path: str = None):
        """Load the model into memory."""
        pass

    @abstractmethod
    def predict(self, image_path: str) -> tuple:
        """Process the image and return (predicted_class_name, confidence_percentage)."""
        pass
