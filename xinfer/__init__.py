"""Top-level package for xinfer."""

__author__ = """Dickson Neoh"""
__email__ = "dickson.neoh@gmail.com"
__version__ = "0.0.11"

from .core import create_model, list_models
from .model_registry import ModelInputOutput, register_model
from .models import BaseModel
from .utils import (
    ollama_available,
    timm_available,
    transformers_available,
    ultralytics_available,
    vllm_available,
)

if timm_available:
    from .timm import *
if transformers_available:
    from .transformers import *
if ultralytics_available:
    from .ultralytics import *
if vllm_available:
    from .vllm import *
if ollama_available:
    from .ollama import *

from .viz import launch_gradio_demo

__all__ = [
    "create_model",
    "list_models",
    "register_model",
    "BaseModel",
    "ModelInputOutput",
    "launch_gradio_demo",
]
