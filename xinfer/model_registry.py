from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Type

from .models import BaseModel


class ModelInputOutput(Enum):
    IMAGE_TO_TEXT = "image --> text"
    IMAGE_TEXT_TO_TEXT = "image-text --> text"
    TEXT_TO_TEXT = "text --> text"
    IMAGE_TO_BOXES = "image --> boxes"
    IMAGE_TO_CATEGORIES = "image --> categories"


@dataclass
class ModelInfo:
    id: str
    implementation: str
    input_output: ModelInputOutput


class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}

    def register(self, model_info: ModelInfo, model_class: Type[BaseModel]):
        if model_info.id in self._models:
            raise ValueError(
                f"Model {model_info.id} already registered. Pick another id."
            )
        self._models[model_info.id] = (model_info, model_class)

    def get_model(self, model_id: str, **kwargs) -> BaseModel:
        model_info, model_class = self._models.get(model_id, (None, None))
        if model_class is None:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_class(model_id, **kwargs)

    def list_models(self) -> List[ModelInfo]:
        return [model_info for model_info, _ in self._models.values()]

    def get_model_info(self, name: str) -> ModelInfo:
        model_info, _ = self._models.get(name, (None, None))
        if model_info is None:
            raise ValueError(f"Unsupported model: {name}")
        return model_info


# Create a global instance of the registry
model_registry = ModelRegistry()


def register_model(model_id: str, implementation: str, input_output: ModelInputOutput):
    def decorator(cls: Type[BaseModel]):
        model_registry.register(ModelInfo(model_id, implementation, input_output), cls)
        return cls

    return decorator
