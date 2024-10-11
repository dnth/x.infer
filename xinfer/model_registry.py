from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Type

from .base_model import BaseModel


class InputOutput(Enum):
    IMAGE_TO_TEXT = "image --> text"
    IMAGE_TEXT_TO_TEXT = "image-text --> text"
    TEXT_TO_TEXT = "text --> text"
    IMAGE_TO_BBOX = "image --> bbox"
    IMAGE_TO_CLASS = "image --> class"


@dataclass
class ModelInfo:
    model_class: Type[BaseModel]
    input_output: InputOutput


@dataclass
class BackendRegistry:
    backend_name: str
    models: Dict[str, ModelInfo] = field(default_factory=dict)


class ModelRegistry:
    _registry: Dict[str, BackendRegistry] = {}

    @classmethod
    def register(
        cls,
        backend: str,
        model_id: str,
        model_class: Type[BaseModel],
        input_output: InputOutput,
    ):
        if backend not in cls._registry:
            cls._registry[backend] = BackendRegistry(backend_name=backend)
        cls._registry[backend].models[model_id] = ModelInfo(
            model_class=model_class, input_output=input_output
        )

    @classmethod
    def get_model(cls, model_id: str, backend: str, **kwargs) -> BaseModel:
        if backend not in cls._registry:
            raise ValueError(f"Unsupported backend: {backend}")
        if model_id not in cls._registry[backend].models:
            raise ValueError(f"Unsupported model type for {backend}: {model_id}")
        return cls._registry[backend].models[model_id].model_class(**kwargs)

    @classmethod
    def list_models(cls) -> List[Dict[str, str]]:
        return [
            {
                "backend": backend,
                "model_id": model_id,
                "input_output": model_info.input_output,
            }
            for backend, backend_registry in cls._registry.items()
            for model_id, model_info in backend_registry.models.items()
        ]
