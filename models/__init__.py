# models/__init__.py

from .base_model import BasePlantModel
from .custom_model import CustomModel
from .juppy_model import JuppyModel
from .juppy_extended_model import JuppyExtendedModel
from .plantnet_model import PlantNetModel
from .plantnet_extended_model import PlantNetExtendedModel

__all__ = [
    "BasePlantModel",
    "CustomModel",
    "JuppyModel",
    "JuppyExtendedModel",
    "PlantNetModel",
    "PlantNetExtendedModel",
    "get_model",
]

MODEL_REGISTRY = {
    "custom": CustomModel,
    "juppy": JuppyModel,
    "juppy_extended": JuppyExtendedModel,
    "plantnet": PlantNetModel,
    "plantnet_extended": PlantNetExtendedModel,
}


def get_model(
    model_key: str, name: str = None, model_path: str = None
) -> BasePlantModel:
    """
    Factory function to instantiate models dynamically.

    Example:
        model = get_model("custom", name="ResNet18", model_path="path/to/resnet18_folder")
    """
    model_key = model_key.lower()
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model key '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[model_key]

    # CustomModel accepts model_path_or_repo
    if model_key == "custom":
        return model_cls(name=name or "Custom Model", model_path_or_repo=model_path)

    return model_cls()
