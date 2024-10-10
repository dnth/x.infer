from typing import Dict, List, Type

from .base_model import BaseModel


class ModelRegistry:
    _registry: Dict[str, Dict[str, Type[BaseModel]]] = {}

    @classmethod
    def register(
        cls, implementation: str, model_type: str, model_class: Type[BaseModel]
    ):
        if implementation not in cls._registry:
            cls._registry[implementation] = {}
        cls._registry[implementation][model_type] = model_class

    @classmethod
    def get_model(cls, model_type: str, implementation: str, **kwargs) -> BaseModel:
        print(cls._registry)
        if implementation not in cls._registry:
            raise ValueError(f"Unsupported implementation: {implementation}")
        if model_type not in cls._registry[implementation]:
            raise ValueError(
                f"Unsupported model type for {implementation}: {model_type}"
            )
        return cls._registry[implementation][model_type](**kwargs)

    @classmethod
    def list_models(cls) -> List[Dict[str, str]]:
        models = []
        for implementation, model_types in cls._registry.items():
            for model_type in model_types:
                models.append(
                    {"implementation": implementation, "model_type": model_type}
                )
        return models
