import difflib
from typing import Dict, List, Type

from .models import BaseXInferModel
from .types import ModelInfo, ModelInputOutput


class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}

    def register(self, model_info: ModelInfo, model_class: Type[BaseXInferModel]):
        if model_info.id in self._models:
            raise ValueError(
                f"Model {model_info.id} already registered. Pick another id."
            )
        self._models[model_info.id] = (model_info, model_class)

    def get_model(self, model_id: str, **kwargs) -> BaseXInferModel:
        model_info, model_class = self._models.get(model_id, (None, None))
        if model_class is None:
            supported_models = list(self._models.keys())
            similar_models = difflib.get_close_matches(
                model_id, supported_models, n=10, cutoff=0.6
            )
            if similar_models:
                suggestion = ", ".join(similar_models)
                raise ValueError(
                    f"Unsupported model: {model_id}. Suggestion model: {suggestion}。"
                )
            else:
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
    def decorator(cls: Type[BaseXInferModel]):
        model_registry.register(ModelInfo(model_id, implementation, input_output), cls)
        return cls

    return decorator
