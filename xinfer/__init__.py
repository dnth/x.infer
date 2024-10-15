"""Top-level package for xinfer."""

__author__ = """Dickson Neoh"""
__email__ = "dickson.neoh@gmail.com"
__version__ = "0.0.3"

import importlib

from .base_model import BaseModel
from .core import create_model, list_models
from .model_registry import ModelInputOutput, register_model


def soft_import(name: str):
    try:
        importlib.import_module(name)
        return True
    except ModuleNotFoundError as e:
        if str(e) != f"No module named '{name}'":
            raise e
        return False


timm_available = soft_import("timm")
transformers_available = soft_import("transformers")
ultralytics_available = soft_import("ultralytics")

if timm_available:
    from .timm import *
if transformers_available:
    from .transformers import *
if ultralytics_available:
    from .ultralytics import *

__all__ = [
    "create_model",
    "list_models",
    "register_model",
    "BaseModel",
    "ModelInputOutput",
]
